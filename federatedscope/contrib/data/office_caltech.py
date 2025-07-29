import os
import logging
import json
import numpy as np
from torchvision import transforms
from torchvision.datasets import ImageFolder, DatasetFolder
from sklearn.cluster import DBSCAN
from scipy.stats import entropy

from federatedscope.core.data import BaseDataTranslator
from federatedscope.register import register_data
from federatedscope.core.data import ClientData, StandaloneDataDict
from federatedscope.contrib.data.utils import (
    dirichlet_split,
    get_label_distribution,
    adaptive_cluster_clients,
    save_cluster_results,
    uniform_split,
    normalize_distributions,
    js_divergence,
    compute_jsd_matrix,
)

logger = logging.getLogger(__name__)


##########################
# 聚类相关函数（参考cross_domain_digits.py）
##########################


# def get_label_distribution(subsets: list, num_classes=10):
#     """
#     计算每个子集的标签分布
#     :param subsets: list of Subset objects or datasets
#     :param num_classes: number of classes in the dataset
#     :return: list of distributions for each subset
#     """
#     distributions = []
#     for subset in subsets:
#         labels = []
#         # 获取标签
#         if hasattr(subset, "dataset"):
#             # 如果是Subset对象
#             for idx in subset.indices:
#                 _, label = subset.dataset[idx]
#                 labels.append(label)
#         else:
#             # 如果是普通dataset
#             for i in range(len(subset)):
#                 _, label = subset[i]
#                 labels.append(label)

#         labels = np.array(labels)
#         dist = np.zeros(num_classes, dtype=int)
#         for label in labels:
#             dist[label] += 1
#         distributions.append(dist)
#     return distributions


# def normalize_distributions(distributions: list):
#     """标准化分布"""
#     normalized = []
#     for dist in distributions:
#         s = np.sum(dist)
#         if s > 0:
#             normalized.append(dist / s)
#         else:
#             normalized.append(dist)
#     return np.array(normalized)


# def js_divergence(p, q):
#     """计算Jensen-Shannon散度"""
#     # 平滑处理避免log(0)
#     p = np.array(p) + 1e-10
#     q = np.array(q) + 1e-10
#     p /= p.sum()
#     q /= q.sum()

#     # 计算混合分布
#     m = 0.5 * (p + q)

#     # JS散度 = (KL(P||M) + KL(Q||M))/2
#     return 0.5 * (entropy(p, m) + entropy(q, m))


# def compute_jsd_matrix(norm_dists):
#     """计算Jensen-Shannon散度距离矩阵"""
#     n = len(norm_dists)
#     jsd_matrix = np.zeros((n, n))

#     for i in range(n):
#         for j in range(i + 1, n):
#             # 计算JSD(P||Q) = √[JSD(P||Q)] (确保度量满足三角不等式)
#             jsd_matrix[i, j] = jsd_matrix[j, i] = np.sqrt(
#                 js_divergence(norm_dists[i], norm_dists[j])
#             )
#     return jsd_matrix


# def adaptive_cluster_clients(
#     train_distributions,
#     target_avg_size=4,  # 每个簇平均包含的用户数
#     min_threshold=0.01,  # 最小阈值
#     max_threshold=1.0,  # 最大阈值
#     max_iter=15,  # 最大迭代次数
#     tolerance=1,  # 簇大小容忍范围
# ):
#     """
#     自适应调整阈值的聚类算法
#     """
#     n_clients = len(train_distributions)
#     target_clusters = max(1, n_clients // target_avg_size)  # 目标簇数
#     norm_dists = normalize_distributions(train_distributions)
#     jsd_matrix = compute_jsd_matrix(norm_dists)

#     logger.info(
#         f"开始自适应聚类: {n_clients}个客户端, 目标平均簇大小: {target_avg_size}, 目标簇数: {target_clusters}"
#     )

#     # 使用二分搜索自动调整阈值
#     low, high = min_threshold, max_threshold
#     best_jsd_threshold = low
#     best_clusters = None

#     for i in range(max_iter):
#         mid_jsd = (low + high) / 2
#         logger.info(
#             f"迭代#{i+1}: 尝试阈值 {mid_jsd:.4f} (范围 [{low:.4f}, {high:.4f}])"
#         )

#         clustering = DBSCAN(
#             eps=mid_jsd, min_samples=1, metric="precomputed"  # 单个样本可成簇
#         )
#         cluster_labels = clustering.fit_predict(jsd_matrix)
#         num_clusters = len(np.unique(cluster_labels))

#         # 计算实际平均簇大小
#         cluster_sizes = [
#             np.sum(cluster_labels == label) for label in np.unique(cluster_labels)
#         ]
#         avg_cluster_size = np.mean(cluster_sizes) if num_clusters > 0 else 0

#         # 记录最佳结果
#         if best_clusters is None or abs(avg_cluster_size - target_avg_size) < tolerance:
#             best_jsd_threshold = mid_jsd
#             best_clusters = cluster_labels
#             best_size = avg_cluster_size
#             best_count = num_clusters

#         logger.info(f"结果: {num_clusters}个簇, 平均大小: {avg_cluster_size:.2f}")

#         # 检查是否达到目标范围
#         if abs(avg_cluster_size - target_avg_size) <= tolerance:
#             logger.info(
#                 f"找到合适阈值: {mid_jsd:.4f}, 簇数: {num_clusters}, 平均大小: {avg_cluster_size:.2f}"
#             )
#             break

#         # 调整搜索范围
#         if avg_cluster_size > target_avg_size:
#             # 簇数过少，需要减少阈值使得簇数更多
#             high = mid_jsd
#         else:
#             # 簇数过多，需要增加阈值使得簇数更少
#             low = mid_jsd

#         # 检查收敛
#         if high - low < 0.001:
#             logger.info(
#                 f"达到收敛: 最佳阈值 {best_jsd_threshold:.4f}, 簇数: {best_count}, 平均大小: {best_size:.2f}"
#             )
#             break
#     else:
#         logger.warning(
#             f"达到最大迭代次数 {max_iter}，使用最佳结果: 阈值 {best_jsd_threshold:.4f}"
#         )

#     # 构建社区结构
#     peer_communities = []
#     for label in np.unique(best_clusters):
#         community = np.where(best_clusters == label)[0]
#         peer_communities.append(community.tolist())

#     # 记录最终簇大小分布
#     cluster_sizes = [len(c) for c in peer_communities]
#     logger.info(
#         f"聚类完成: {len(peer_communities)}个簇, 平均大小: {np.mean(cluster_sizes):.2f}, "
#         f"最小: {np.min(cluster_sizes)}, 最大: {np.max(cluster_sizes)}"
#     )

#     return peer_communities


# def save_cluster_results(filename, train_dists, test_dists, communities):
#     """保存聚类结果到JSON文件"""
#     # 准备可JSON序列化的数据结构
#     data = {
#         "metadata": {
#             "num_clients": len(train_dists),
#             "num_classes": len(train_dists[0]),
#             "num_communities": len(communities),
#         },
#         "train_distributions": [
#             dist.tolist() if isinstance(dist, np.ndarray) else dist
#             for dist in train_dists
#         ],
#         "test_distributions": [
#             dist.tolist() if isinstance(dist, np.ndarray) else dist
#             for dist in test_dists
#         ],
#         "peer_communities": [
#             community.tolist() if isinstance(community, np.ndarray) else community
#             for community in communities
#         ],
#     }

#     # 写入文件
#     with open(filename, "w") as f:
#         json.dump(data, f, indent=2)

#     logger.info(f"聚类结果保存成功！文件: {filename}")


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
        subset_train_num=7,
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
    config, client_cfgs, train_dataset, test_dataset, domain_name
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
    tmp_config.merge_from_list(["federate.client_num", 10])

    # Use BaseDataTranslator to convert to federated format
    translator = BaseDataTranslator(tmp_config, client_cfgs)

    # Create dataset tuple (train, val, test) - using test for both val and test
    split_datasets = (train_dataset, test_dataset, test_dataset)
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
    from torch.utils.data import ConcatDataset

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
    for dict_idx, data_dict in enumerate(data_dicts):
        client_id_offset = dict_idx * 10  # 0, 10, 20, 30
        for client_id in range(1, 11):
            new_client_id = client_id + client_id_offset
            merged_data_dict[new_client_id] = data_dict[client_id]
            # Update client configuration to merged config
            merged_data_dict[new_client_id].setup(tmp_config_merged)

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

    for domain_name in domain_names:
        # Load domain dataset
        train_dataset, test_dataset = _load_domain_dataset(
            domain_name, data_root, train_transform, test_transform
        )

        # Create federated data dictionary for this domain
        data_dict = _create_federated_data_dict(
            config, client_cfgs, train_dataset, test_dataset, domain_name
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
