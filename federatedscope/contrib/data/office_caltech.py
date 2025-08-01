import os
import logging
import json
import numpy as np
from torchvision import transforms
from torchvision.datasets import ImageFolder, DatasetFolder
from sklearn.cluster import DBSCAN
from scipy.stats import entropy
from federatedscope.core.auxiliaries.splitter_builder import get_splitter
from federatedscope.core.data import BaseDataTranslator
from federatedscope.register import register_data
from federatedscope.core.data import ClientData, StandaloneDataDict
from federatedscope.contrib.data.utils import (
    get_label_distribution,
    adaptive_cluster_clients,
    save_cluster_results,
    uniform_split,
)
from torch.utils.data import ConcatDataset, Subset
import numpy as np

logger = logging.getLogger(__name__)


class ImageFolder_Custom(DatasetFolder):
    """
    Custom ImageFolder implementation based on RethinkFL's officecaltech.py
    Supports train/test split based on subset sampling for Caltech dataset
    """

    def __init__(
        self,
        data_name,
        root,
        train=True,
        transform=None,
        target_transform=None,
        subset_train_num=4,
        subset_capacity=10,
    ):
        self.data_name = data_name
        self.root = root
        self.train = train
        self.transform = transform
        self.target_transform = target_transform

        # Build the path to the Caltech dataset
        dataset_path = os.path.join(self.root, "Office_Caltech_10", self.data_name)
        if not os.path.exists(dataset_path):
            # Fallback to direct domain name path
            dataset_path = os.path.join(self.root, self.data_name)

        self.imagefolder_obj = ImageFolder(
            dataset_path, self.transform, self.target_transform
        )

        all_data = self.imagefolder_obj.samples
        self.train_index_list = []
        self.test_index_list = []

        # Split data based on subset sampling strategy from RethinkFL
        for i in range(len(all_data)):
            if i % subset_capacity <= subset_train_num:
                self.train_index_list.append(i)
            else:
                self.test_index_list.append(i)

    def __len__(self):
        if self.train:
            return len(self.train_index_list)
        else:
            return len(self.test_index_list)

    def __getitem__(self, index):
        if self.train:
            used_index_list = self.train_index_list
        else:
            used_index_list = self.test_index_list

        path = self.imagefolder_obj.samples[used_index_list[index]][0]
        target = self.imagefolder_obj.samples[used_index_list[index]][1]
        target = int(target)
        img = self.imagefolder_obj.loader(path)

        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


def _load_domain_dataset(data_name, data_root, train_transform, test_transform):
    """
    Load a single domain dataset (caltech, amazon, webcam, or dslr)

    Args:
        data_name: Name of the domain dataset
        data_root: Root directory for data
        train_transform: Transform for training data
        test_transform: Transform for test data

    Returns:
        tuple: (train_dataset, test_dataset)
    """
    try:
        # Create train dataset
        train_dataset = ImageFolder_Custom(
            data_name=data_name, root=data_root, train=True, transform=train_transform
        )

        # Create test dataset
        test_dataset = ImageFolder_Custom(
            data_name=data_name, root=data_root, train=False, transform=test_transform
        )

        logger.info(
            f"Loaded {data_name} dataset: train={len(train_dataset)}, test={len(test_dataset)}"
        )
        return train_dataset, test_dataset

    except Exception as e:
        logger.error(f"Failed to load {data_name} dataset: {e}")
        raise e


def _create_federated_data_dict(
    config,
    client_cfgs,
    train_dataset,
    test_dataset,
    domain_name,
    client_num=10,
    classes_per_client=2,
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
        dict: Federated data dictionary with 10 clients
    """
    # Create temporary config for this domain (10 clients)
    tmp_config = config.clone()
    tmp_config.merge_from_list(["federate.client_num", client_num])

    # Use BaseDataTranslator to convert to federated format
    # translator = BaseDataTranslator(tmp_config, client_cfgs)
    splitter = get_splitter(tmp_config)
    if splitter is None:
        raise ValueError(f"Splitter not found for {domain_name}")
    train_data_lists = splitter(train_dataset, classes_per_client=classes_per_client)
    test_data_lists = splitter(test_dataset, classes_per_client=classes_per_client)
    # test_data_indices = uniform_split(test_dataset, client_num)
    # test_data_lists = [Subset(test_dataset, indices) for indices in test_data_indices]
    data_dict = {
        client_id: ClientData(
            tmp_config,
            train=train_data_lists[client_id - 1],
            val=test_data_lists[client_id - 1],
            test=test_data_lists[client_id - 1],
        )
        for client_id in range(1, client_num + 1)
    }
    data_dict[0] = ClientData(
        tmp_config,
        train=train_dataset,
        val=test_dataset,
        test=test_dataset,
    )  # Server data

    logger.info(
        f"Successfully created federated {domain_name} dataset with {len(data_dict)} clients"
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

    total_clients = len(data_dicts) * 10  # Each dict has 10 clients

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
    # 输出每个客户端训练集和测试集的大小
    for client_id, client_data in merged_data_dict.items():
        train_size = len(client_data.train_data) if client_data.train_data else 0
        test_size = len(client_data.test_data) if client_data.test_data else 0
        logger.info(
            f"Client {client_id}: train size={train_size}, test size={test_size}"
        )
    # Create final StandaloneDataDict
    return StandaloneDataDict(merged_data_dict, tmp_config_merged)


def _perform_clustering_analysis(data_dict, data_root):
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
            f"收集到 {len(train_subsets)} 个训练子集, {len(test_subsets)} 个测试子集"
        )

        if train_subsets and test_subsets:
            # Calculate label distributions
            logger.info("计算客户端标签分布...")
            train_distributions = get_label_distribution(train_subsets, num_classes=10)
            test_distributions = get_label_distribution(test_subsets, num_classes=10)

            # Perform clustering
            logger.info("执行客户端聚类...")
            peer_communities = adaptive_cluster_clients(train_distributions)

            logger.info(f"聚类结果: {peer_communities}")

            # Save clustering results
            cluster_file = os.path.join(data_root, "peer_communities.json")
            save_cluster_results(
                cluster_file, train_distributions, test_distributions, peer_communities
            )

        else:
            logger.warning("无法获取客户端数据子集，跳过聚类步骤")

    except Exception as e:
        logger.error(f"聚类过程中出现错误: {e}")
        logger.info("继续返回联邦数据，但没有聚类信息")


def load_office_caltech_dataset(config, client_cfgs=None):
    """
    Load Office Caltech dataset for federated learning based on RethinkFL implementation

    Args:
        config: Configuration object containing data settings
        client_cfgs: Client-specific configurations (optional)

    Returns:
        tuple: (datadict, config) containing federated dataset
    """
    data_root = config.data.root

    # Define data transforms from RethinkFL
    train_transform = transforms.Compose(
        [
            transforms.Resize((32, 32)),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )

    test_transform = transforms.Compose(
        [
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )

    # Load all domain datasets
    domain_names = ["caltech", "amazon", "webcam", "dslr"]
    data_dicts = []
    # client_nums = [8, 8, 4, 4]
    clients_per_domain = [9, 9, 1, 1]
    classes_per_client_per_domain = [2, 2, 10, 10]

    for domain_name, client_num, classes_per_client in zip(
        domain_names, clients_per_domain, classes_per_client_per_domain
    ):
        # Load domain dataset
        train_dataset, test_dataset = _load_domain_dataset(
            domain_name, data_root, train_transform, test_transform
        )

        # Create federated data dictionary for this domain
        data_dict = _create_federated_data_dict(
            config,
            client_cfgs,
            train_dataset,
            test_dataset,
            domain_name,
            client_num,
            classes_per_client,
        )
        data_dicts.append(data_dict)

    # Merge all data dictionaries
    merged_data_dict = _merge_data_dictionaries(config, data_dicts)

    # Perform clustering analysis
    _perform_clustering_analysis(merged_data_dict, data_root)

    return merged_data_dict, config


def call_caltech_data(config, client_cfgs=None):
    """
    Entry point for Caltech dataset registration.

    Args:
        config: Configuration object
        client_cfgs: Client-specific configurations (optional)

    Returns:
        tuple: (datadict, config) if dataset type matches, None otherwise
    """
    if config.data.type.lower() == "office_caltech":
        logger.info("Loading Caltech dataset...")
        return load_office_caltech_dataset(config, client_cfgs)
    return None


# Register the Caltech dataset
register_data("office_caltech", call_caltech_data)
