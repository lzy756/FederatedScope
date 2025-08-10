import os
import logging
import numpy as np
import random
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, Subset, ConcatDataset
from federatedscope.core.data import BaseDataTranslator
from federatedscope.register import register_data
from federatedscope.core.data import ClientData, StandaloneDataDict
from federatedscope.contrib.data.utils import (
    get_label_distribution,
    adaptive_cluster_clients,
    save_cluster_results,
)

logger = logging.getLogger(__name__)

# DomainNet domains
USED_DOMAINS = [
    "clipart",
    "infograph",
    "painting",
    "quickdraw",
    "real",
    "sketch",
]


# Extract common classes across all domains
def load_common_classes(base_path, domains):
    domain_classes = []

    logger.info(f"Looking for common classes in domains: {domains}")
    logger.info(f"Base path: {base_path}")

    # Collect class sets for each domain
    for domain in domains:
        domain_path = os.path.join(base_path, domain)
        if not os.path.exists(domain_path):
            logger.warning(f"Domain {domain} does not exist at {domain_path}")
            continue

        # Get all classes (directory names) for this domain
        try:
            all_items = os.listdir(domain_path)
            # Only keep directories (classes)
            classes = set(
                [
                    item
                    for item in all_items
                    if os.path.isdir(os.path.join(domain_path, item))
                ]
            )
            logger.info(f"Domain {domain} has {len(classes)} classes")
            domain_classes.append(classes)
        except Exception as e:
            logger.error(f"Error reading domain {domain}: {e}")
            continue

    # Calculate intersection of all domain classes
    if domain_classes:
        common_classes = set.intersection(*domain_classes)
        logger.info(f"Found {len(common_classes)} common classes across all domains")
        if len(common_classes) == 0:
            logger.warning("No common classes found across all domains!")
        return common_classes
    else:
        logger.error("No valid domains found")
        return set()


def count_class_samples(base_path, domains, classes):
    """
    Count the total number of samples for each class across all domains

    Args:
        base_path: Root path to the dataset
        domains: List of domain names
        classes: List of class names to count

    Returns:
        Dict mapping class names to total sample counts
    """
    class_counts = {cls_name: 0 for cls_name in classes}

    for domain in domains:
        for split in ["train", "test"]:
            split_file = os.path.join(base_path, f"{domain}_{split}.txt")
            if not os.path.exists(split_file):
                continue

            try:
                with open(split_file, "r") as f:
                    lines = f.readlines()

                for line in lines:
                    line = line.strip()
                    if not line:
                        continue

                    try:
                        path, _ = line.split(" ", 1)
                        class_name = os.path.basename(os.path.dirname(path))

                        if class_name in classes:
                            # Check if image file exists
                            img_path = os.path.join(base_path, path)
                            if os.path.exists(img_path):
                                class_counts[class_name] += 1
                    except ValueError:
                        continue

            except Exception as e:
                logger.warning(f"Error reading {split_file}: {e}")
                continue

    return class_counts


def select_classes_by_sample_count(base_path, num_classes, selection_method="least"):
    """
    Select classes based on their sample counts across all domains

    Args:
        base_path: Root path to the dataset
        num_classes: Number of classes to select
        selection_method: "most" for most samples, "least" for least samples, "random" for random selection

    Returns:
        List of selected class names
    """
    all_classes = load_common_classes(base_path, USED_DOMAINS)

    if len(all_classes) == 0:
        raise ValueError("No common classes found across domains!")

    if num_classes > len(all_classes):
        logger.warning(
            f"Requested {num_classes} classes but only {len(all_classes)} available. Using all available classes."
        )
        return sorted(list(all_classes))

    # Count samples for each class
    logger.info("Counting samples for each class across all domains...")
    class_counts = count_class_samples(base_path, USED_DOMAINS, all_classes)

    # Sort classes by sample count
    if selection_method == "most":
        # Select classes with most samples (descending order)
        sorted_classes = sorted(class_counts.items(), key=lambda x: x[1], reverse=True)
        logger.info("Selecting classes with the most samples...")
    elif selection_method == "least":
        # Select classes with least samples (ascending order)
        sorted_classes = sorted(class_counts.items(), key=lambda x: x[1], reverse=False)
        logger.info("Selecting classes with the least samples...")
    else:  # random
        # Random selection (for backward compatibility)
        random.seed(42)
        selected_classes = sorted(random.sample(list(all_classes), num_classes))
        logger.info(
            f"Randomly selected {len(selected_classes)} classes: {selected_classes}"
        )
        return selected_classes

    # Select top num_classes
    selected_classes = [cls_name for cls_name, count in sorted_classes[:num_classes]]

    # Log selection results
    logger.info(
        f"Selected {len(selected_classes)} classes by {selection_method} sample count:"
    )
    for cls_name in selected_classes:
        count = class_counts[cls_name]
        logger.info(f"  {cls_name}: {count} samples")

    return selected_classes


# Keep original function for backward compatibility
def random_select_classes(base_path, num_classes, seed=42):
    """
    Randomly select a specified number of classes from common classes
    (Kept for backward compatibility)

    Args:
        base_path: Root path to the dataset
        num_classes: Number of classes to select
        seed: Random seed for reproducibility

    Returns:
        List of selected class names
    """
    return select_classes_by_sample_count(base_path, num_classes, "random")


class DomainNetFullImages:
    """
    Helper class to load all images from a specific domain and classes
    """

    def __init__(self, domain_name, root_dir, class_list):
        self.domain_name = domain_name
        self.root_dir = root_dir
        self.class_list = class_list

        logger.info(
            f"Loading all data for domain '{domain_name}' with {len(class_list)} classes: {class_list}"
        )

        # Load data from both train and test splits
        self.all_data = []
        self.class_to_idx = {}

        # Create class mapping
        sorted_classes = sorted(class_list)
        self.class_to_idx = {
            cls_name: idx for idx, cls_name in enumerate(sorted_classes)
        }
        self.classes = sorted_classes
        self.num_classes = len(sorted_classes)

        # Load from both train and test splits
        for split in ["train", "test"]:
            split_file = os.path.join(root_dir, f"{domain_name}_{split}.txt")
            if not os.path.exists(split_file):
                logger.warning(f"Split file not found: {split_file}")
                continue

            logger.info(f"Loading from {split_file}")
            with open(split_file, "r") as f:
                lines = f.readlines()

            split_count = 0
            for line in lines:
                line = line.strip()
                if not line:
                    continue

                try:
                    path, _ = line.split(" ", 1)  # Only split on first space
                    class_name = os.path.basename(os.path.dirname(path))

                    if class_name in class_list:
                        img_path = os.path.join(root_dir, path)
                        if os.path.exists(img_path):
                            label = self.class_to_idx[class_name]
                            self.all_data.append((img_path, label))
                            split_count += 1
                        else:
                            logger.debug(f"Image file not found: {img_path}")
                except ValueError as e:
                    logger.warning(f"Invalid line format: {line}")
                    continue

            logger.info(f"Loaded {split_count} samples from {split}")

        if not self.all_data:
            raise ValueError(
                f"No data found for specified classes in domain {domain_name}"
            )

        logger.info(
            f"Loaded {len(self.all_data)} total samples for domain {domain_name}"
        )

    def __len__(self):
        return len(self.all_data)


class DomainNetDataset(Dataset):
    """
    Custom DomainNet dataset implementation for train/test splits
    """

    def __init__(self, data_list, class_to_idx, classes, num_classes, transform=None):
        """
        Initialize DomainNet dataset with specific data list

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
        img_path, label = self.all_data[index]

        try:
            image = Image.open(img_path).convert("RGB")
        except (IOError, OSError) as e:
            logger.error(f"Unable to open image: {img_path}, error: {e}")
            return None

        if self.transform is not None:
            image = self.transform(image)

        return image, label


def _load_domain_dataset(
    domain_name,
    data_root,
    train_transform,
    test_transform,
    sampled_classes,
    train_ratio,
):
    """
    Load a single domain dataset and split into train/test sets.

    Args:
        domain_name: Name of the domain dataset
        data_root: Root directory for data
        train_transform: Transform for training data
        test_transform: Transform for test data
        sampled_classes: List of selected classes
        train_ratio: Ratio of data to use for training

    Returns:
        tuple: (train_dataset, test_dataset)
    """
    try:
        # Load all data first using DomainNetFullImages
        full_images = DomainNetFullImages(
            domain_name=domain_name, root_dir=data_root, class_list=sampled_classes
        )

        # Sample data if needed to limit size
        all_data = full_images.all_data
        if len(all_data) > 12000:  # Total limit before splitting
            np.random.seed(42)  # For reproducibility
            indices = np.random.choice(len(all_data), 12000, replace=False)
            all_data = [all_data[i] for i in indices]
            logger.info(
                f"Sampled {len(all_data)} samples from total {len(full_images.all_data)}"
            )

        # Randomly shuffle data indices
        indices = np.arange(len(all_data))
        np.random.shuffle(indices)

        # Split data into train and test sets
        train_size = int(len(all_data) * train_ratio)
        train_indices = indices[:train_size]
        test_indices = indices[train_size:]

        # Create train data list
        train_data = [all_data[i] for i in train_indices]

        # Create test data list
        test_data = [all_data[i] for i in test_indices]

        # Create independent train dataset
        train_dataset = DomainNetDataset(
            data_list=train_data,
            class_to_idx=full_images.class_to_idx,
            classes=full_images.classes,
            num_classes=full_images.num_classes,
            transform=train_transform,
        )

        # Create independent test dataset
        test_dataset = DomainNetDataset(
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

    Returns:
        dict: Federated data dictionary
    """
    # Create temporary config for this domain
    tmp_config = config.clone()
    tmp_config.merge_from_list(["federate.client_num", client_num])

    # Use BaseDataTranslator to convert to federated format
    translator = BaseDataTranslator(tmp_config, client_cfgs)

    # Create dataset tuple (train, val, test) - using test for both val and test
    split_datasets = (train_dataset, [], test_dataset)
    data_dict = translator(split_datasets)

    logger.info(
        f"Successfully created federated {domain_name} dataset with {len(data_dict) - 1} clients"
    )
    return data_dict


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
    server_val = ConcatDataset([data_dict[0].val_data for data_dict in data_dicts])
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
            merged_data_dict[cnt] = ClientData(
                tmp_config_merged,
                train=client_data.train_data,
                val=client_data.val_data,
                test=client_data.test_data,
            )
            cnt += 1

    logger.info(f"Merged data dictionary created with {len(merged_data_dict)} clients")
    # Log the size of training and test sets for each client
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


def load_domainnet_dataset(config, client_cfgs=None):
    """
    Load DomainNet dataset for federated learning.

    Args:
        config: Configuration object containing data settings
        client_cfgs: Client-specific configurations (optional)

    Returns:
        tuple: (datadict, config) containing federated dataset
    """
    data_root = config.data.root
    num_classes = config.model.out_channels

    if num_classes <= 0:
        # Default to using all common classes
        all_common = load_common_classes(data_root, USED_DOMAINS)
        num_classes = len(all_common)
        logger.info(f"Using all {num_classes} common classes by default")

    # Select classes based on sample count (default: least samples first)
    sampled_classes = select_classes_by_sample_count(data_root, num_classes, "least")

    # Define data transforms
    transform_train = transforms.Compose(
        [
            transforms.Resize([256, 256]),
            transforms.RandomCrop((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation((-30, 30)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    transform_test = transforms.Compose(
        [
            transforms.Resize([256, 256]),
            transforms.CenterCrop((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # Load all domain datasets
    domain_names = USED_DOMAINS
    clients_per_domain = config.data.clients_per_domain
    train_ratio = config.data.splits[0]

    data_dicts = []

    for domain_name, client_num in zip(domain_names, clients_per_domain):
        # Load domain dataset
        train_dataset, test_dataset = _load_domain_dataset(
            domain_name,
            data_root,
            transform_train,
            transform_test,
            sampled_classes,
            train_ratio,
        )

        # Create federated data dictionary for this domain
        data_dict = _create_federated_data_dict(
            config,
            client_cfgs,
            train_dataset,
            test_dataset,
            domain_name,
            client_num,
        )
        data_dicts.append(data_dict)

    # Merge all data dictionaries
    merged_data_dict = _merge_data_dictionaries(config, data_dicts)

    # Perform clustering analysis
    _perform_clustering_analysis(merged_data_dict, data_root, num_classes)

    return merged_data_dict, config


def call_domainnet_data(config, client_cfgs=None):
    """
    Entry point for DomainNet dataset registration.

    Args:
        config: Configuration object
        client_cfgs: Client-specific configurations (optional)

    Returns:
        tuple: (datadict, config) if dataset type matches, None otherwise
    """
    if config.data.type.lower() == "domainnet":
        logger.info("Loading DomainNet dataset...")
        return load_domainnet_dataset(config, client_cfgs)
    return None


# Register the DomainNet dataset
register_data("domainnet", call_domainnet_data)
