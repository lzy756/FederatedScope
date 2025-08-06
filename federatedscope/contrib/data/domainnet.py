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
    sample_data,
)

logger = logging.getLogger(__name__)

# DomainNet domains
DOMAINNET_DOMAINS = ["clipart", "infograph", "painting", "quickdraw", "real", "sketch"]
USED_DOMAINS = [
    "clipart",
    "painting",
    "quickdraw",
    "infograph",
    "sketch",
]  # Currently used domains


# 取出所有域的共有类别
def load_common_classes(base_path, domains):
    domain_classes = []

    # 收集每个域的类别集合
    for domain in domains:
        domain_path = os.path.join(base_path, domain)
        if not os.path.exists(domain_path):
            logger.warning(f"Domain {domain} does not exist at {domain_path}")
            continue
        # 获取该域的所有类别（目录名）
        classes = set(os.listdir(domain_path))
        domain_classes.append(classes)

    # 计算所有域类别的交集
    if domain_classes:
        common_classes = set.intersection(*domain_classes)
        logger.info(f"Common classes across domains: {common_classes}")
        return common_classes
    else:
        logger.error("No valid domains found")
        return set()


# 随机选择指定数量的类
def random_select_classes(base_path, num_classes):
    all_classes = load_common_classes(base_path, DOMAINNET_DOMAINS)
    if num_classes > len(all_classes):
        return list(all_classes)
    selected_classes = random.sample(all_classes, num_classes)
    return selected_classes


class DomainNetDatasetFull(Dataset):
    """
    Custom DomainNet implementation that loads all data from both train and test splits
    and allows manual splitting later (similar to PACS ImageFolder_Custom)
    """

    def __init__(self, domain_name, root_dir, class_list, transform=None):
        self.domain_name = domain_name
        self.root_dir = root_dir
        self.transform = transform
        self.class_list = class_list

        logger.info(
            f"Loading all data for domain '{domain_name}' with classes: {class_list}"
        )

        # Load data from both train and test splits
        self.all_data = []
        self.class_to_idx = {}

        # 创建类别映射
        sorted_classes = sorted(class_list)
        self.class_to_idx = {
            cls_name: idx for idx, cls_name in enumerate(sorted_classes)
        }

        # Load from both train and test splits
        for split in ["train", "test"]:
            split_file = os.path.join(root_dir, f"{domain_name}_{split}.txt")
            if not os.path.exists(split_file):
                logger.warning(f"Split file not found: {split_file}")
                continue

            with open(split_file, "r") as f:
                lines = f.readlines()

            for line in lines:
                path, _ = line.strip().split(" ")
                class_name = os.path.basename(os.path.dirname(path))

                if class_name in class_list:
                    img_path = os.path.join(root_dir, path)
                    label = self.class_to_idx[class_name]
                    self.all_data.append((img_path, label))

        if not self.all_data:
            raise ValueError(f"在域 {domain_name} 中未找到指定类别的数据")

        logger.info(
            f"Loaded {len(self.all_data)} total samples for domain {domain_name}"
        )

    def __len__(self):
        return len(self.all_data)

    def __getitem__(self, index):
        img_path, label = self.all_data[index]

        try:
            image = Image.open(img_path).convert("RGB")
        except:
            logger.error(f"无法打开图像: {img_path}")
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
    train_ratio=0.4,
):
    """
    Load a single domain dataset and split into train/test sets.
    Similar to PACS implementation - load all data first, then split.

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
        # Load all data first (similar to PACS ImageFolder_Custom)
        full_dataset = DomainNetDatasetFull(
            domain_name=domain_name, root_dir=data_root, class_list=sampled_classes
        )

        # Sample data if needed to limit size
        if len(full_dataset) > 12000:  # Total limit before splitting
            full_dataset = sample_data(full_dataset, 12000)

        # Randomly shuffle data indices (similar to PACS)
        all_data = full_dataset.all_data
        indices = np.arange(len(all_data))
        np.random.shuffle(indices)

        # Split data into train and test sets
        train_size = int(len(all_data) * train_ratio)
        train_indices = indices[:train_size]
        test_indices = indices[train_size:]

        # Create train dataset with train transform
        train_dataset = Subset(full_dataset, train_indices)
        train_dataset.dataset.transform = train_transform

        # Create test dataset with test transform
        test_dataset = Subset(full_dataset, test_indices)
        test_dataset.dataset.transform = test_transform

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
        num_classes: Number of classes
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
            train_distributions = get_label_distribution(
                train_subsets, num_classes=num_classes
            )
            test_distributions = get_label_distribution(
                test_subsets, num_classes=num_classes
            )

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
    total_clients = config.federate.client_num
    classes = config.model.out_channels
    sampled_classes = random_select_classes(data_root, classes)

    # Define data transforms
    transform_train = transforms.Compose(
        [
            transforms.Resize([256, 256]),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation((-30, 30)),
            transforms.ToTensor(),
        ]
    )
    transform_test = transforms.Compose(
        [
            transforms.Resize([256, 256]),
            transforms.ToTensor(),
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
    _perform_clustering_analysis(merged_data_dict, data_root, len(sampled_classes))

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
