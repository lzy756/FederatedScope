import os
import logging
import numpy as np
from torchvision import transforms
from torchvision.datasets import ImageFolder
from federatedscope.core.data import BaseDataTranslator
from federatedscope.register import register_data
from federatedscope.core.data import ClientData, StandaloneDataDict
from federatedscope.contrib.data.utils import (
    get_label_distribution,
    adaptive_cluster_clients,
    save_cluster_results,
)
from torch.utils.data import ConcatDataset, Subset, Dataset

logger = logging.getLogger(__name__)


class OfficeFullImages:
    """
    Helper class to load all images from a specific domain and classes for Office Caltech
    """

    def __init__(self, domain_name, root):
        self.domain_name = domain_name
        self.root = root

        logger.info(f"Loading all data for domain '{domain_name}'")

        # Build the path to the Office Caltech dataset
        dataset_path = os.path.join(self.root, "office_caltech_10", self.domain_name)
        if not os.path.exists(dataset_path):
            # Fallback to direct domain name path
            dataset_path = os.path.join(self.root, self.domain_name)

        # Use ImageFolder to get all samples
        from torchvision.datasets import ImageFolder

        imagefolder_obj = ImageFolder(dataset_path, transform=None)
        self.all_data = imagefolder_obj.samples

        # Store class information and standardize class names
        original_classes = imagefolder_obj.classes
        original_class_to_idx = imagefolder_obj.class_to_idx

        # Standardize class names: 'back_pack' -> 'backpack'
        self.classes = []
        self.class_to_idx = {}
        standardized_samples = []

        # Create mapping from original to standardized class names
        class_name_mapping = {}
        for i, class_name in enumerate(original_classes):
            standardized_name = class_name.replace("back_pack", "backpack")
            self.classes.append(standardized_name)
            self.class_to_idx[standardized_name] = i
            class_name_mapping[original_class_to_idx[class_name]] = i

        # Update samples with standardized class indices
        for sample_path, original_label in self.all_data:
            standardized_label = class_name_mapping[original_label]
            standardized_samples.append((sample_path, standardized_label))

        self.all_data = standardized_samples
        self.num_classes = len(self.classes)

        logger.info(
            f"Loaded {len(self.all_data)} total samples for domain {domain_name}"
        )
        logger.info(f"Classes: {self.classes}")

    def __len__(self):
        return len(self.all_data)


class OfficeDataset(Dataset):
    """
    Custom Office Caltech dataset implementation for train/test splits
    """

    def __init__(self, data_list, class_to_idx, classes, num_classes, transform=None):
        """
        Initialize Office dataset with specific data list

        Args:
            data_list: List of (img_path, label) tuples
            class_to_idx: Dictionary mapping class names to indices
            classes: List of class names
            num_classes: Number of classes
            transform: Transform to apply to images
        """
        self.all_data = data_list
        self.class_to_idx = class_to_idx
        self.classes = classes
        self.num_classes = num_classes
        self.transform = transform

    def __len__(self):
        return len(self.all_data)

    def __getitem__(self, index):
        path, target = self.all_data[index]
        target = int(target)

        # Load image using PIL
        from PIL import Image

        img = Image.open(path).convert("RGB")

        if self.transform is not None:
            img = self.transform(img)

        return img, target


def _load_domain_dataset(
    domain_name, data_root, train_transform, test_transform, train_ratio=0.4
):
    """
    Load a single domain dataset (caltech, amazon, webcam, or dslr) and split into train/test sets.

    Args:
        domain_name: Name of the domain dataset
        data_root: Root directory for data
        train_transform: Transform for training data
        test_transform: Transform for test data
        train_ratio: Ratio of data to use for training

    Returns:
        tuple: (train_dataset, test_dataset)
    """
    try:
        # Load all data first using OfficeFullImages
        full_images = OfficeFullImages(domain_name=domain_name, root=data_root)

        logger.info(
            f"Loading {domain_name} dataset from {data_root}: classes={full_images.classes}, num_classes={full_images.num_classes}, class_to_idx={full_images.class_to_idx}"
        )

        # Get all data
        all_data = full_images.all_data

        # Group data by class to ensure stratified splitting
        class_data = {}
        for i, (_, label) in enumerate(all_data):
            if label not in class_data:
                class_data[label] = []
            class_data[label].append(i)

        # Stratified split: ensure each class is represented in both train and test
        train_indices = []
        test_indices = []

        for _, indices in class_data.items():
            # Shuffle indices for this class
            class_indices = np.array(indices)
            np.random.shuffle(class_indices)

            # Split this class according to train_ratio
            class_train_size = int(len(class_indices) * train_ratio)
            # Ensure at least one sample in each split if possible
            if class_train_size == 0 and len(class_indices) > 1:
                class_train_size = 1
            elif class_train_size == len(class_indices) and len(class_indices) > 1:
                class_train_size = len(class_indices) - 1

            class_train_indices = class_indices[:class_train_size]
            class_test_indices = class_indices[class_train_size:]

            train_indices.extend(class_train_indices)
            test_indices.extend(class_test_indices)

        # Convert to numpy arrays and shuffle the final indices
        train_indices = np.array(train_indices)
        test_indices = np.array(test_indices)
        np.random.shuffle(train_indices)
        np.random.shuffle(test_indices)

        # Create train data list
        train_data = [all_data[i] for i in train_indices]

        # Create test data list
        test_data = [all_data[i] for i in test_indices]

        # Create independent train dataset
        train_dataset = OfficeDataset(
            data_list=train_data,
            class_to_idx=full_images.class_to_idx,
            classes=full_images.classes,
            num_classes=full_images.num_classes,
            transform=train_transform,
        )

        # Create independent test dataset
        test_dataset = OfficeDataset(
            data_list=test_data,
            class_to_idx=full_images.class_to_idx,
            classes=full_images.classes,
            num_classes=full_images.num_classes,
            transform=test_transform,
        )

        logger.info(
            f"Loaded {domain_name} dataset: train={len(train_dataset)}, test={len(test_dataset)}"
        )
        return train_dataset, test_dataset

    except Exception as e:
        logger.error(f"Failed to load {domain_name} dataset: {e}")
        raise e


def _create_federated_data_dict(
    config,
    client_cfgs,
    train_dataset,
    test_dataset,
    domain_name,
    client_num,
    val_dataset=None,
):
    """
    Create federated data dictionary for a single domain dataset

    Args:
        config: Original configuration object
        client_cfgs: Client-specific configurations
        train_dataset: Training dataset
        test_dataset: Test dataset
        domain_name: Name of the domain for logging
        client_num: Number of clients for this domain
        val_dataset: Validation dataset (optional)

    Returns:
        dict: Federated data dictionary with client_num clients
    """
    # Create temporary config for this domain (client_num clients)
    tmp_config = config.clone()
    tmp_config.merge_from_list(["federate.client_num", client_num])

    # Use BaseDataTranslator to convert to federated format
    translator = BaseDataTranslator(tmp_config, client_cfgs)

    # Create dataset tuple (train, val, test)
    if val_dataset is None:
        val_dataset = []
    split_datasets = (train_dataset, val_dataset, test_dataset)
    data_dict = translator(split_datasets)

    logger.info(
        f"Successfully created federated {domain_name} dataset with {len(data_dict) - 1} clients"
    )
    return data_dict


def _create_cross_domain_validation_sets(
    all_domain_datasets, clients_per_domain, target_val_per_client=50
):
    """
    Create cross-domain validation sets for each domain.
    Each domain's validation set contains samples from other domains' test sets.
    The validation set size is determined by the number of clients in the domain to ensure fair distribution.

    Args:
        all_domain_datasets: Dictionary mapping domain names to their datasets
        clients_per_domain: List of client numbers for each domain
        target_val_per_client: Target number of validation samples per client

    Returns:
        dict: Dictionary mapping domain names to their cross-domain validation datasets
    """
    domain_names = list(all_domain_datasets.keys())
    cross_domain_val_sets = {}

    for domain_idx, target_domain in enumerate(domain_names):
        # Calculate target validation set size based on number of clients in this domain
        num_clients_in_domain = clients_per_domain[domain_idx]
        target_val_size = num_clients_in_domain * target_val_per_client

        logger.info(
            f"Target domain '{target_domain}' has {num_clients_in_domain} clients, "
            f"targeting {target_val_size} validation samples ({target_val_per_client} per client)"
        )

        # Collect all available samples from other domains' test sets
        other_domain_class_data = {}

        for source_domain in domain_names:
            if source_domain != target_domain:
                source_test_dataset = all_domain_datasets[source_domain]["test"]
                source_all_data = source_test_dataset.all_data

                # Group data by class for stratified sampling
                for sample_idx, (img_path, label) in enumerate(source_all_data):
                    if label not in other_domain_class_data:
                        other_domain_class_data[label] = []
                    other_domain_class_data[label].append((img_path, label))

        # Stratified sampling to get target_val_size samples while maintaining class balance
        other_domain_samples = []
        if other_domain_class_data:
            # Calculate samples per class
            num_classes = len(other_domain_class_data)
            samples_per_class = max(1, target_val_size // num_classes)

            for class_label, samples in other_domain_class_data.items():
                # Shuffle samples for this class
                class_samples = np.array(samples, dtype=object)
                np.random.shuffle(class_samples)

                # Take up to samples_per_class samples, allow repetition if not enough
                if len(class_samples) >= samples_per_class:
                    # Enough samples, no repetition needed
                    selected_samples = class_samples[:samples_per_class]
                else:
                    # Not enough samples, repeat samples to reach target
                    selected_samples = []
                    samples_needed = samples_per_class
                    while samples_needed > 0:
                        if samples_needed >= len(class_samples):
                            # Need more than available, take all and repeat
                            selected_samples.extend(class_samples)
                            samples_needed -= len(class_samples)
                        else:
                            # Need partial, take what's needed
                            selected_samples.extend(class_samples[:samples_needed])
                            samples_needed = 0

                        # Reshuffle for next round to avoid patterns
                        if samples_needed > 0:
                            np.random.shuffle(class_samples)

                # Add selected samples to validation set
                other_domain_samples.extend(selected_samples)

            # If we still need more samples after stratified sampling, add more with repetition
            if len(other_domain_samples) < target_val_size:
                # Collect all available samples and allow repetition
                all_available_samples = []
                for class_label, samples in other_domain_class_data.items():
                    all_available_samples.extend(samples)

                if all_available_samples:
                    additional_needed = target_val_size - len(other_domain_samples)

                    # Repeat samples if necessary to meet the target
                    repeated_samples = []
                    while len(repeated_samples) < additional_needed:
                        # Shuffle to avoid patterns
                        np.random.shuffle(all_available_samples)

                        # Add samples, up to what we need
                        remaining_needed = additional_needed - len(repeated_samples)
                        if remaining_needed >= len(all_available_samples):
                            repeated_samples.extend(all_available_samples)
                        else:
                            repeated_samples.extend(
                                all_available_samples[:remaining_needed]
                            )

                    other_domain_samples.extend(repeated_samples)

                    logger.info(
                        f"Domain '{target_domain}': Used sample repetition to reach target size. "
                        f"Total unique samples available: {len(all_available_samples)}, "
                        f"Target size: {target_val_size}"
                    )

        # Create validation dataset from collected samples
        if other_domain_samples:
            val_dataset = OfficeDataset(
                data_list=other_domain_samples,
                class_to_idx=all_domain_datasets[target_domain]["train"].class_to_idx,
                classes=all_domain_datasets[target_domain]["train"].classes,
                num_classes=all_domain_datasets[target_domain]["train"].num_classes,
                transform=all_domain_datasets[target_domain][
                    "test"
                ].transform,  # Use test transform for validation
            )
            cross_domain_val_sets[target_domain] = val_dataset
            logger.info(
                f"Created cross-domain validation set for {target_domain}: {len(val_dataset)} samples "
                f"({len(val_dataset) / num_clients_in_domain:.1f} samples per client)"
            )
        else:
            cross_domain_val_sets[target_domain] = None
            logger.warning(
                f"No cross-domain validation samples created for {target_domain}"
            )

    return cross_domain_val_sets


def _merge_data_dictionaries(config, data_dicts):
    """
    Merge multiple federated data dictionaries into a single one

    Args:
        config: Original configuration object
        data_dicts: List of federated data dictionaries to merge

    Returns:
        StandaloneDataDict: Merged federated data dictionary
    """

    total_clients = sum(len(data_dict) for data_dict in data_dicts) - len(data_dicts)

    # Create merged config
    tmp_config_merged = config.clone()
    tmp_config_merged.merge_from_list(["federate.client_num", total_clients])

    # Create merged data dictionary
    merged_data_dict = {}

    # Merge server data (ID=0)
    server_train = ConcatDataset([data_dict[0].train_data for data_dict in data_dicts])
    # For server validation, concatenate all client validation sets if they exist
    server_val_datasets = []
    for data_dict in data_dicts:
        if (
            hasattr(data_dict[0], "val_data")
            and data_dict[0].val_data is not None
            and len(data_dict[0].val_data) > 0
        ):
            server_val_datasets.append(data_dict[0].val_data)
    server_val = (
        ConcatDataset(server_val_datasets) if server_val_datasets else ConcatDataset([])
    )

    server_test = ConcatDataset([data_dict[0].test_data for data_dict in data_dicts])
    merged_data_dict[0] = ClientData(
        tmp_config_merged, train=server_train, val=server_val, test=server_test
    )

    # Add clients from each data dictionary with proper ID offset
    cnt = 1
    for data_dict in data_dicts:
        for client_id, client_data in data_dict.items():
            if client_id == 0:
                continue
            if cnt not in merged_data_dict:
                merged_data_dict[cnt] = ClientData(
                    tmp_config_merged,
                    train=client_data.train_data,
                    val=client_data.val_data,
                    test=client_data.test_data,
                )
            cnt += 1

    logger.info(f"Merged data dictionary created with {len(merged_data_dict)} clients")
    # Log the size of training, validation and test sets for each client
    for client_id, client_data in merged_data_dict.items():
        train_size = len(client_data.train_data) if client_data.train_data else 0
        val_size = len(client_data.val_data) if client_data.val_data else 0
        test_size = len(client_data.test_data) if client_data.test_data else 0
        logger.info(
            f"Client {client_id}: train size={train_size}, val size={val_size}, test size={test_size}"
        )
    # Create final StandaloneDataDict
    return StandaloneDataDict(merged_data_dict, tmp_config_merged)


def _perform_clustering_analysis(data_dict, data_root, num_classes):
    """
    Perform clustering analysis on federated data

    Args:
        data_dict: Federated data dictionary
        data_root: Root directory for saving results
    """
    try:
        # Collect training and test subsets
        train_subsets = []
        test_subsets = []

        # Extract data from each client (skip server id=0)
        for client_id in range(1, len(data_dict)):
            client_data = data_dict[client_id]

            # Access original datasets rather than DataLoaders
            if (
                hasattr(client_data, "train_data")
                and client_data.train_data is not None
            ):
                train_subsets.append(client_data.train_data)
            if hasattr(client_data, "test_data") and client_data.test_data is not None:
                test_subsets.append(client_data.test_data)

        logger.info(
            f"Collected {len(train_subsets)} training subsets, {len(test_subsets)} test subsets"
        )

        if train_subsets and test_subsets:
            # Calculate label distributions
            logger.info("Calculating client label distributions...")
            train_distributions = get_label_distribution(train_subsets, num_classes)
            test_distributions = get_label_distribution(test_subsets, num_classes)

            # Perform clustering
            logger.info("Performing client clustering...")
            peer_communities = adaptive_cluster_clients(train_distributions)

            logger.info(f"Clustering results: {peer_communities}")

            # Save clustering results
            cluster_file = os.path.join(data_root, "peer_communities.json")
            save_cluster_results(
                cluster_file, train_distributions, test_distributions, peer_communities
            )

        else:
            logger.warning(
                "Unable to obtain client data subsets, skipping clustering step"
            )

    except Exception as e:
        logger.error(f"Error occurred during clustering process: {e}")
        logger.info(
            "Continuing to return federated data, but without clustering information"
        )


def load_office_caltech_dataset(config, client_cfgs=None):
    """
    Load Office Caltech dataset for federated learning.

    Args:
        config: Configuration object containing data settings
        client_cfgs: Client-specific configurations (optional)

    Returns:
        tuple: (datadict, config) containing federated dataset
    """
    data_root = config.data.root
    num_classes = config.model.out_channels
    # Define data transforms
    transform_train = transforms.Compose(
        [
            transforms.Resize((256, 256)),  # Resize to 256x256 first
            transforms.RandomCrop((224, 224)),  # Random crop to 224x224
            transforms.RandomHorizontalFlip(p=0.5),  # Random horizontal flip
            transforms.ToTensor(),  # Convert to tensor
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),  # Normalization
        ]
    )

    transform_test = transforms.Compose(
        [
            transforms.Resize((256, 256)),  # Resize to 256x256
            transforms.CenterCrop(
                (224, 224)
            ),  # Center crop to avoid random factors affecting evaluation
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # Load all domain datasets first
    domain_names = ["caltech", "amazon", "webcam", "dslr"]
    clients_per_domain = config.data.clients_per_domain
    train_ratio = config.data.splits[0]

    # Step 1: Load all domain datasets
    all_domain_datasets = {}
    for domain_name in domain_names:
        train_dataset, test_dataset = _load_domain_dataset(
            domain_name, data_root, transform_train, transform_test, train_ratio
        )
        all_domain_datasets[domain_name] = {
            "train": train_dataset,
            "test": test_dataset,
        }

    # Step 2: Create cross-domain validation sets
    logger.info("Creating cross-domain validation sets...")
    # Get target validation samples per client from config
    cross_domain_val_sets = _create_cross_domain_validation_sets(
        all_domain_datasets, clients_per_domain, 200
    )

    # Step 3: Create federated data dictionaries with cross-domain validation
    data_dicts = []
    for domain_name, client_num in zip(domain_names, clients_per_domain):
        train_dataset = all_domain_datasets[domain_name]["train"]
        test_dataset = all_domain_datasets[domain_name]["test"]
        val_dataset = cross_domain_val_sets[domain_name]

        # Create federated data dictionary for this domain with cross-domain validation
        data_dict = _create_federated_data_dict(
            config,
            client_cfgs,
            train_dataset,
            test_dataset,
            domain_name,
            client_num,
            val_dataset,
        )
        data_dicts.append(data_dict)

    # Merge all data dictionaries
    merged_data_dict = _merge_data_dictionaries(config, data_dicts)

    # Perform clustering analysis
    _perform_clustering_analysis(merged_data_dict, data_root, num_classes)

    return merged_data_dict, config


def call_office_caltech_data(config, client_cfgs=None):
    """
    Entry point for Office Caltech dataset registration.

    Args:
        config: Configuration object
        client_cfgs: Client-specific configurations (optional)

    Returns:
        tuple: (datadict, config) if dataset type matches, None otherwise
    """
    if config.data.type.lower() == "office_caltech":
        logger.info("Loading Office Caltech dataset...")
        return load_office_caltech_dataset(config, client_cfgs)
    return None


# Register the Office Caltech dataset
register_data("office_caltech", call_office_caltech_data)
