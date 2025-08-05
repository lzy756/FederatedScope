import os
import logging
import numpy as np
from torchvision import transforms
from torchvision.datasets import ImageFolder, DatasetFolder

from federatedscope.core.data import BaseDataTranslator
from federatedscope.register import register_data
from federatedscope.core.data import ClientData, StandaloneDataDict
from federatedscope.contrib.splitter.fedgs_splitter import FedGS_Splitter

logger = logging.getLogger(__name__)


##########################
# 聚类相关函数（使用FedGS_Splitter）
##########################

def get_label_distribution(subsets: list, num_classes=10):
    """
    计算每个子集的标签分布
    :param subsets: list of Subset objects or datasets
    :param num_classes: number of classes in the dataset
    :return: list of distributions for each subset
    """
    distributions = []
    for subset in subsets:
        labels = []
        # 获取标签
        if hasattr(subset, 'dataset'):
            # 如果是Subset对象
            for idx in subset.indices:
                _, label = subset.dataset[idx]
                labels.append(label)
        else:
            # 如果是普通dataset
            for i in range(len(subset)):
                _, label = subset[i]
                labels.append(label)

        labels = np.array(labels)
        dist = np.zeros(num_classes, dtype=int)
        for label in labels:
            dist[label] += 1
        distributions.append(dist)
    return distributions


def perform_clustering_with_fedgs(train_distributions,
                                  distance_threshold=0.5,
                                  min_groups=2,
                                  max_groups=10,
                                  balance_weight=2.0,
                                  balance_tolerance=0.2):
    """
    使用FedGS_Splitter进行聚类

    Args:
        train_distributions: 客户端训练数据分布列表
        distance_threshold: 距离阈值
        min_groups: 最小簇数量
        max_groups: 最大簇数量
        balance_weight: 大小均衡权重
        balance_tolerance: 大小均衡容忍度

    Returns:
        list: 聚类结果（peer communities）
    """
    n_clients = len(train_distributions)

    # 创建FedGS_Splitter实例
    splitter = FedGS_Splitter(
        client_num=n_clients,
        distance_threshold=distance_threshold,
        min_groups=min_groups,
        max_groups=max_groups,
        balance_weight=balance_weight,
        balance_tolerance=balance_tolerance
    )

    # 设置数据分布信息
    splitter.data_info = []
    for dist in train_distributions:
        splitter.data_info.append({"distribution": np.array(dist)})

    # 执行聚类
    logger.info(f"使用FedGS_Splitter进行聚类: {n_clients}个客户端")
    splitter.build_peer_communities(train_distributions)

    return splitter.peer_communities


class ImageFolder_Custom(DatasetFolder):
    """
    Custom ImageFolder implementation based on RethinkFL's officecaltech.py
    Supports train/test split based on subset sampling for Caltech dataset
    """
    def __init__(self, data_name, root, train=True, transform=None, target_transform=None,
                 subset_train_num=7, subset_capacity=10):
        self.data_name = data_name
        self.root = root
        self.train = train
        self.transform = transform
        self.target_transform = target_transform

        # Build the path to the Caltech dataset
        dataset_path = os.path.join(self.root, 'Office_Caltech_10', self.data_name)
        if not os.path.exists(dataset_path):
            # Fallback to direct domain name path
            dataset_path = os.path.join(self.root, self.data_name)

        self.imagefolder_obj = ImageFolder(dataset_path, self.transform, self.target_transform)

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
            data_name=data_name,
            root=data_root,
            train=True,
            transform=train_transform
        )

        # Create test dataset
        test_dataset = ImageFolder_Custom(
            data_name=data_name,
            root=data_root,
            train=False,
            transform=test_transform
        )

        logger.info(f"Loaded {data_name} dataset: train={len(train_dataset)}, test={len(test_dataset)}")
        return train_dataset, test_dataset

    except Exception as e:
        logger.error(f"Failed to load {data_name} dataset: {e}")
        raise e


def _create_federated_data_dict(config, client_cfgs, train_dataset, test_dataset, domain_name, domain_client_num):
    """
    Create federated data dictionary for a single domain dataset

    Args:
        config: Original configuration object
        client_cfgs: Client-specific configurations
        train_dataset: Training dataset
        test_dataset: Test dataset
        domain_name: Name of the domain for logging
        domain_client_num: Number of clients for this domain

    Returns:
        dict: Federated data dictionary with specified number of clients
    """
    # Create temporary config for this domain
    tmp_config = config.clone()
    tmp_config.merge_from_list(['federate.client_num', domain_client_num])

    # Use BaseDataTranslator to convert to federated format
    translator = BaseDataTranslator(tmp_config, client_cfgs)

    # Create dataset tuple (train, val, test) - using test for both val and test
    split_datasets = (train_dataset, test_dataset, test_dataset)
    data_dict = translator(split_datasets)

    logger.info(f"Successfully created federated {domain_name} dataset with {len(data_dict) - 1} clients")
    return data_dict


def _merge_data_dictionaries(config, data_dicts, domain_client_nums):
    """
    Merge multiple federated data dictionaries into a single one

    Args:
        config: Original configuration object
        data_dicts: List of federated data dictionaries to merge
        domain_client_nums: List of client numbers for each domain

    Returns:
        StandaloneDataDict: Merged federated data dictionary
    """
    from torch.utils.data import ConcatDataset

    total_clients = sum(domain_client_nums)

    # Create merged config
    tmp_config_merged = config.clone()
    tmp_config_merged.merge_from_list(['federate.client_num', total_clients])

    # Create merged data dictionary
    merged_data_dict = {}

    # Merge server data (ID=0)
    server_train = ConcatDataset([data_dict[0].train_data for data_dict in data_dicts])
    server_val = ConcatDataset([data_dict[0].val_data for data_dict in data_dicts])
    server_test = ConcatDataset([data_dict[0].test_data for data_dict in data_dicts])
    merged_data_dict[0] = ClientData(tmp_config_merged, train=server_train, val=server_val, test=server_test)

    # Add clients from each data dictionary with proper ID offset
    client_id_offset = 0
    for data_dict, domain_client_num in zip(data_dicts, domain_client_nums):
        for client_id in range(1, domain_client_num + 1):
            new_client_id = client_id + client_id_offset
            merged_data_dict[new_client_id] = data_dict[client_id]
            # Update client configuration to merged config
            merged_data_dict[new_client_id].setup(tmp_config_merged)

        # Update offset for next domain
        client_id_offset += domain_client_num

    # Create final StandaloneDataDict
    return StandaloneDataDict(merged_data_dict, tmp_config_merged)


def _perform_clustering_analysis(data_dict, data_root):
    """
    Perform clustering analysis on federated data using FedGS_Splitter

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
            if hasattr(client_data, 'train_data') and client_data.train_data is not None:
                train_subsets.append(client_data.train_data)
            if hasattr(client_data, 'test_data') and client_data.test_data is not None:
                test_subsets.append(client_data.test_data)

        logger.info(f"收集到 {len(train_subsets)} 个训练子集, {len(test_subsets)} 个测试子集")

        if train_subsets and test_subsets:
            # Calculate label distributions
            logger.info("计算客户端标签分布...")
            train_distributions = get_label_distribution(train_subsets, num_classes=10)
            test_distributions = get_label_distribution(test_subsets, num_classes=10)

            # Perform clustering using FedGS_Splitter
            logger.info("使用FedGS_Splitter执行客户端聚类...")
            peer_communities = perform_clustering_with_fedgs(
                train_distributions,
                distance_threshold=0.5,
                min_groups=2,
                max_groups=10,
                balance_weight=2.0,
                balance_tolerance=0.2
            )

            logger.info(f"聚类结果: {peer_communities}")

            # Save clustering results using FedGS_Splitter's save method
            n_clients = len(train_distributions)
            splitter = FedGS_Splitter(client_num=n_clients)

            # Set the data and communities for saving
            splitter.data_info = [{"distribution": np.array(dist)} for dist in train_distributions]
            splitter.test_data_info = [{"distribution": np.array(dist)} for dist in test_distributions]
            splitter.peer_communities = peer_communities

            # Save to the specified location
            cluster_file = os.path.join(data_root, "peer_communities.json")
            splitter.save_cluster_results(cluster_file)

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
    total_clients = config.federate.client_num

    # Define data transforms from RethinkFL
    train_transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    test_transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    # Load all domain datasets and calculate sample counts
    domain_names = ['caltech', 'amazon', 'webcam', 'dslr']
    domain_datasets = []
    domain_sample_counts = []

    for domain_name in domain_names:
        # Load domain dataset
        train_dataset, test_dataset = _load_domain_dataset(
            domain_name, data_root, train_transform, test_transform
        )
        domain_datasets.append((train_dataset, test_dataset))

        # Count samples in this domain (using training samples for proportion calculation)
        sample_count = len(train_dataset)
        domain_sample_counts.append(sample_count)
        logger.info(f"Domain {domain_name}: {sample_count} training samples")

    # Calculate client allocation based on sample proportions
    total_samples = sum(domain_sample_counts)
    domain_client_nums = []
    allocated_clients = 0

    for i, sample_count in enumerate(domain_sample_counts):
        if i == len(domain_sample_counts) - 1:
            # Last domain gets remaining clients
            domain_client_num = total_clients - allocated_clients
        else:
            # Calculate proportional allocation
            proportion = sample_count / total_samples
            domain_client_num = max(1, round(proportion * total_clients))  # At least 1 client per domain
            allocated_clients += domain_client_num

        domain_client_nums.append(domain_client_num)
        logger.info(f"Domain {domain_names[i]}: {domain_client_num} clients "
                   f"(proportion: {sample_count/total_samples:.3f})")

    # Verify total allocation
    actual_total = sum(domain_client_nums)
    if actual_total != total_clients:
        logger.warning(f"Client allocation mismatch: expected {total_clients}, got {actual_total}")
        # Adjust the last domain to match exactly
        domain_client_nums[-1] += (total_clients - actual_total)
        logger.info(f"Adjusted last domain to {domain_client_nums[-1]} clients")

    # Create federated data dictionaries for each domain
    data_dicts = []
    for i, (domain_name, (train_dataset, test_dataset), domain_client_num) in enumerate(
            zip(domain_names, domain_datasets, domain_client_nums)):

        # Create federated data dictionary for this domain
        data_dict = _create_federated_data_dict(
            config, client_cfgs, train_dataset, test_dataset, domain_name, domain_client_num
        )
        data_dicts.append(data_dict)

    # Merge all data dictionaries
    merged_data_dict = _merge_data_dictionaries(config, data_dicts, domain_client_nums)

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