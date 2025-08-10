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


class PacsFullImages:
    """
    Helper class to load all images from a specific domain for PACS dataset
    """

    def __init__(self, domain_name, root):
        self.domain_name = domain_name
        self.root = root

        logger.info(f"Loading all data for domain '{domain_name}'")

        # Build the path to the PACS dataset
        dataset_path = os.path.join(self.root, "PACS", self.domain_name)
        if not os.path.exists(dataset_path):
            # Fallback to direct domain name path
            dataset_path = os.path.join(self.root, self.domain_name)

        # Use ImageFolder to get all samples
        imagefolder_obj = ImageFolder(dataset_path, transform=None)
        self.all_data = imagefolder_obj.samples

        # Store class information
        self.classes = imagefolder_obj.classes  # List of class names
        self.class_to_idx = (
            imagefolder_obj.class_to_idx
        )  # Dict mapping class names to indices
        self.num_classes = len(self.classes)  # Number of classes

        logger.info(
            f"Loaded {len(self.all_data)} total samples for domain {domain_name}"
        )
        logger.info(f"Classes: {self.classes}")

    def __len__(self):
        return len(self.all_data)


class PacsDataset(Dataset):
    """
    Custom PACS dataset implementation for train/test splits
    """

    def __init__(self, data_list, class_to_idx, classes, num_classes, transform=None):
        """
        Initialize PACS dataset with specific data list

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
    Load a single domain dataset and split into train/test sets.

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
        # Load all data first using PacsFullImages
        full_images = PacsFullImages(domain_name=domain_name, root=data_root)

        logger.info(
            f"Loading {domain_name} dataset from {data_root}: classes={full_images.classes}, num_classes={full_images.num_classes}, class_to_idx={full_images.class_to_idx}"
        )

        # Get all data
        all_data = full_images.all_data

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
        train_dataset = PacsDataset(
            data_list=train_data,
            class_to_idx=full_images.class_to_idx,
            classes=full_images.classes,
            num_classes=full_images.num_classes,
            transform=train_transform,
        )

        # Create independent test dataset
        test_dataset = PacsDataset(
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

    Returns:
        dict: Federated data dictionary with client_num clients
    """
    # Create temporary config for this domain (client_num clients)
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
            if cnt not in merged_data_dict:
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


def load_pacs_dataset(config, client_cfgs=None):
    """
    Load PACS dataset for federated learning.

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

    # Load all domain datasets
    domain_names = ["art_painting", "cartoon", "photo", "sketch"]
    clients_per_domain = config.data.clients_per_domain
    train_ratio = config.data.splits[0]

    data_dicts = []

    for domain_name, client_num in zip(domain_names, clients_per_domain):
        # Load domain dataset
        train_dataset, test_dataset = _load_domain_dataset(
            domain_name, data_root, transform_train, transform_test, train_ratio
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


def call_pacs_data(config, client_cfgs=None):
    """
    Entry point for PACS dataset registration.

    Args:
        config: Configuration object
        client_cfgs: Client-specific configurations (optional)

    Returns:
        tuple: (datadict, config) if dataset type matches, None otherwise
    """
    if config.data.type.lower() == "pacs":
        logger.info("Loading PACS dataset...")
        return load_pacs_dataset(config, client_cfgs)
    return None


# Register the PACS dataset
register_data("pacs", call_pacs_data)
