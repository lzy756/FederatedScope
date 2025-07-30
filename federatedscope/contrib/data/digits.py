import os
import numpy as np
from PIL import Image
from sklearn.datasets import load_svmlight_file
import scipy.io as sio
from torch.utils.data import Dataset, TensorDataset, Subset
from torchvision import datasets, transforms
from federatedscope.core.data import BaseDataTranslator
import json
import logging
import random
from sklearn.cluster import DBSCAN
from scipy.stats import entropy
from federatedscope.register import register_data
from federatedscope.core.data import ClientData, StandaloneDataDict
logger = logging.getLogger(__name__)

##########################
# 聚类相关函数（参考cross_domain_digits.py）
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


def normalize_distributions(distributions: list):
    """标准化分布"""
    normalized = []
    for dist in distributions:
        s = np.sum(dist)
        if s > 0:
            normalized.append(dist / s)
        else:
            normalized.append(dist)
    return np.array(normalized)


def js_divergence(p, q):
    """计算Jensen-Shannon散度"""
    # 平滑处理避免log(0)
    p = np.array(p) + 1e-10
    q = np.array(q) + 1e-10
    p /= p.sum()
    q /= q.sum()

    # 计算混合分布
    m = 0.5 * (p + q)

    # JS散度 = (KL(P||M) + KL(Q||M))/2
    return 0.5 * (entropy(p, m) + entropy(q, m))


def compute_jsd_matrix(norm_dists):
    """计算Jensen-Shannon散度距离矩阵"""
    n = len(norm_dists)
    jsd_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(i + 1, n):
            # 计算JSD(P||Q) = √[JSD(P||Q)] (确保度量满足三角不等式)
            jsd_matrix[i, j] = jsd_matrix[j, i] = np.sqrt(
                js_divergence(norm_dists[i], norm_dists[j])
            )
    return jsd_matrix


def adaptive_cluster_clients(
    train_distributions,
    target_avg_size=4,  # 每个簇平均包含的用户数
    min_threshold=0.01,  # 最小阈值
    max_threshold=1.0,  # 最大阈值
    max_iter=15,  # 最大迭代次数
    tolerance=1,  # 簇大小容忍范围
):
    """
    自适应调整阈值的聚类算法
    """
    n_clients = len(train_distributions)
    target_clusters = max(1, n_clients // target_avg_size)  # 目标簇数
    norm_dists = normalize_distributions(train_distributions)
    jsd_matrix = compute_jsd_matrix(norm_dists)

    logger.info(
        f"开始自适应聚类: {n_clients}个客户端, 目标平均簇大小: {target_avg_size}, 目标簇数: {target_clusters}"
    )

    # 使用二分搜索自动调整阈值
    low, high = min_threshold, max_threshold
    best_jsd_threshold = low
    best_clusters = None

    for i in range(max_iter):
        mid_jsd = (low + high) / 2
        logger.info(
            f"迭代#{i+1}: 尝试阈值 {mid_jsd:.4f} (范围 [{low:.4f}, {high:.4f}])"
        )

        clustering = DBSCAN(
            eps=mid_jsd, min_samples=1, metric="precomputed"  # 单个样本可成簇
        )
        cluster_labels = clustering.fit_predict(jsd_matrix)
        num_clusters = len(np.unique(cluster_labels))

        # 计算实际平均簇大小
        cluster_sizes = [
            np.sum(cluster_labels == label) for label in np.unique(cluster_labels)
        ]
        avg_cluster_size = np.mean(cluster_sizes) if num_clusters > 0 else 0

        # 记录最佳结果
        if best_clusters is None or abs(avg_cluster_size - target_avg_size) < tolerance:
            best_jsd_threshold = mid_jsd
            best_clusters = cluster_labels
            best_size = avg_cluster_size
            best_count = num_clusters

        logger.info(f"结果: {num_clusters}个簇, 平均大小: {avg_cluster_size:.2f}")

        # 检查是否达到目标范围
        if abs(avg_cluster_size - target_avg_size) <= tolerance:
            logger.info(
                f"找到合适阈值: {mid_jsd:.4f}, 簇数: {num_clusters}, 平均大小: {avg_cluster_size:.2f}"
            )
            break

        # 调整搜索范围
        if avg_cluster_size > target_avg_size:
            # 簇数过少，需要减少阈值使得簇数更多
            high = mid_jsd
        else:
            # 簇数过多，需要增加阈值使得簇数更少
            low = mid_jsd

        # 检查收敛
        if high - low < 0.001:
            logger.info(
                f"达到收敛: 最佳阈值 {best_jsd_threshold:.4f}, 簇数: {best_count}, 平均大小: {best_size:.2f}"
            )
            break
    else:
        logger.warning(
            f"达到最大迭代次数 {max_iter}，使用最佳结果: 阈值 {best_jsd_threshold:.4f}"
        )

    # 构建社区结构
    peer_communities = []
    for label in np.unique(best_clusters):
        community = np.where(best_clusters == label)[0]
        peer_communities.append(community.tolist())

    # 记录最终簇大小分布
    cluster_sizes = [len(c) for c in peer_communities]
    logger.info(
        f"聚类完成: {len(peer_communities)}个簇, 平均大小: {np.mean(cluster_sizes):.2f}, "
        f"最小: {np.min(cluster_sizes)}, 最大: {np.max(cluster_sizes)}"
    )

    return peer_communities


def save_cluster_results(filename, train_dists, test_dists, communities):
    """保存聚类结果到JSON文件"""
    # 准备可JSON序列化的数据结构
    data = {
        "metadata": {
            "num_clients": len(train_dists),
            "num_classes": len(train_dists[0]),
            "num_communities": len(communities),
        },
        "train_distributions": [
            dist.tolist() if isinstance(dist, np.ndarray) else dist
            for dist in train_dists
        ],
        "test_distributions": [
            dist.tolist() if isinstance(dist, np.ndarray) else dist
            for dist in test_dists
        ],
        "peer_communities": [
            community.tolist() if isinstance(community, np.ndarray) else community
            for community in communities
        ],
    }

    # 写入文件
    with open(filename, "w") as f:
        json.dump(data, f, indent=2)

    logger.info(f"聚类结果保存成功！文件: {filename}")

##########################
# 数据采样函数
##########################

def sample_dataset(dataset, fraction):
    """
    从数据集中随机采样指定比例的数据

    Args:
        dataset: 原始数据集
        fraction: 采样比例 (0, 1]

    Returns:
        Subset: 采样后的数据集子集
    """
    if fraction >= 1.0:
        return dataset

    total_size = len(dataset)
    sample_size = int(total_size * fraction)

    # 生成随机索引
    indices = list(range(total_size))
    random.shuffle(indices)
    sampled_indices = indices[:sample_size]

    logger.info(f"从 {total_size} 个样本中采样 {sample_size} 个样本 (比例: {fraction:.3f})")

    return Subset(dataset, sampled_indices)

##########################
# 数据加载部分（示例）
##########################

def load_usps(data_root, transform):
    usps_dir = os.path.join(data_root, 'USPS')
    X_tr_sp, y_tr = load_svmlight_file(os.path.join(usps_dir, 'usps'), n_features=256)
    X_te_sp, y_te = load_svmlight_file(os.path.join(usps_dir, 'usps.t'), n_features=256)
    X_tr = X_tr_sp.toarray().reshape(-1, 16, 16).astype('uint8')
    X_te = X_te_sp.toarray().reshape(-1, 16, 16).astype('uint8')
    y_tr = y_tr.astype(int) - 1
    y_te = y_te.astype(int) - 1

    class USPSDataset(TensorDataset):
        def __init__(self, X, y, transform):
            self.X, self.y, self.transform = X, y, transform
        def __len__(self):
            return len(self.y)
        def __getitem__(self, idx):
            img = Image.fromarray(self.X[idx], mode='L')
            img = self.transform(img)
            return img, int(self.y[idx])

    return USPSDataset(X_tr, y_tr, transform), USPSDataset(X_te, y_te, transform)

def load_svhn(data_root, transform):
    def load_mat_svhn(split):
        mat = sio.loadmat(os.path.join(data_root, f"SVHN/{split}_32x32.mat"))
        X = np.transpose(mat['X'], (3, 0, 1, 2))
        y = mat['y'].reshape(-1).astype(int)
        y[y == 10] = 0
        class SVHNDataset(Dataset):
            def __init__(self, X, y, transform):
                self.X, self.y, self.transform = X, y, transform
            def __len__(self):
                return len(self.y)
            def __getitem__(self, idx):
                img = Image.fromarray(self.X[idx])
                img = self.transform(img)
                return img, int(self.y[idx])
        return SVHNDataset(X, y, transform)
    return load_mat_svhn('train'), load_mat_svhn('test')

def load_syn(data_root, transform):
    from torchvision.datasets import ImageFolder
    syn_root = os.path.join(data_root, 'SYN')
    train_ds = ImageFolder(root=os.path.join(syn_root, 'imgs_train'), transform=transform)
    test_ds = ImageFolder(root=os.path.join(syn_root, 'imgs_valid'), transform=transform)
    return train_ds, test_ds

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

    # Calculate total clients dynamically
    total_clients = sum(len(data_dict) - 1 for data_dict in data_dicts)  # -1 to exclude server

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
    current_client_id = 1
    for data_dict in data_dicts:
        num_clients_in_dict = len(data_dict) - 1  # Exclude server (ID=0)
        for client_id in range(1, num_clients_in_dict + 1):
            merged_data_dict[current_client_id] = data_dict[client_id]
            # Update client configuration to merged config
            merged_data_dict[current_client_id].setup(tmp_config_merged)
            current_client_id += 1

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

            # Perform clustering
            logger.info("执行客户端聚类...")
            peer_communities = adaptive_cluster_clients(train_distributions)

            logger.info(f"聚类结果: {peer_communities}")

            # Save clustering results
            cluster_file = os.path.join(data_root, "peer_communities.json")
            save_cluster_results(
                cluster_file,
                train_distributions,
                test_distributions,
                peer_communities
            )

        else:
            logger.warning("无法获取客户端数据子集，跳过聚类步骤")

    except Exception as e:
        logger.error(f"聚类过程中出现错误: {e}")
        logger.info("继续返回联邦数据，但没有聚类信息")

##########################
# 主入口函数，注册给FederatedScope调用
##########################

from federatedscope.core.data.base_data import StandaloneDataDict, ClientData
from federatedscope.register import register_data

def load_digits_data(config, client_cfgs=None):
    data_root = config.data.root

    # 设置随机种子以确保可重现性
    random.seed(42)
    np.random.seed(42)

    # 统一的客户端数量
    unified_client_num = 25

    # 加载数据集
    transform = transforms.Compose(
        [
            transforms.Resize((32, 32)),  # 统一缩放到32x32
            # 将灰度图复制为3通道（如果是RGB则无变化）
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),  # 转为Tensor并归一化到[0,1]
            transforms.Normalize(
                mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]
            ),  # 标准化到[-1,1]
        ]
    )

    # MNIST数据集 - 采样1/6的数据
    mnist_train_full = datasets.MNIST(root=data_root, train=True, transform=transform, download=True)
    mnist_test_full = datasets.MNIST(root=data_root, train=False, transform=transform, download=True)
    logger.info(f"Original MNIST dataset: train={len(mnist_train_full)}, test={len(mnist_test_full)}")

    # 对MNIST进行采样
    mnist_train = sample_dataset(mnist_train_full, 1/6)
    mnist_test = sample_dataset(mnist_test_full, 1/6)
    logger.info(f"Sampled MNIST dataset: train={len(mnist_train)}, test={len(mnist_test)}")

    mnist_split_datasets = (mnist_train, mnist_test, mnist_test)
    tmp_config = config.clone()
    tmp_config.merge_from_list(['federate.client_num', unified_client_num])
    translator = BaseDataTranslator(tmp_config, client_cfgs)
    mnist_data_dict = translator(mnist_split_datasets)
    logger.info(f"Successfully created federated MNIST dataset with {len(mnist_data_dict) - 1} clients")

    # USPS数据集 - 保持原始大小
    usps_train, usps_test = load_usps(data_root, transform)
    logger.info(f"Loaded USPS dataset: train={len(usps_train)}, test={len(usps_test)}")
    usps_split_datasets = (usps_train, usps_test, usps_test)
    tmp_config = config.clone()
    tmp_config.merge_from_list(['federate.client_num', unified_client_num])
    translator = BaseDataTranslator(tmp_config, client_cfgs)
    usps_data_dict = translator(usps_split_datasets)
    logger.info(f"Successfully created federated USPS dataset with {len(usps_data_dict) - 1} clients")

    # SVHN数据集 - 采样1/7的数据
    svhn_train_full, svhn_test_full = load_svhn(data_root, transform)
    logger.info(f"Original SVHN dataset: train={len(svhn_train_full)}, test={len(svhn_test_full)}")

    # 对SVHN进行采样
    svhn_train = sample_dataset(svhn_train_full, 1/7)
    svhn_test = sample_dataset(svhn_test_full, 1/7)
    logger.info(f"Sampled SVHN dataset: train={len(svhn_train)}, test={len(svhn_test)}")

    svhn_split_datasets = (svhn_train, svhn_test, svhn_test)
    tmp_config = config.clone()
    tmp_config.merge_from_list(['federate.client_num', unified_client_num])
    translator = BaseDataTranslator(tmp_config, client_cfgs)
    svhn_data_dict = translator(svhn_split_datasets)
    logger.info(f"Successfully created federated SVHN dataset with {len(svhn_data_dict) - 1} clients")
    
    # SYN数据集 - 保持原始大小
    syn_train, syn_test = load_syn(data_root, transform)
    logger.info(f"Loaded SYN dataset: train={len(syn_train)}, test={len(syn_test)}")
    syn_split_datasets = (syn_train, syn_test, syn_test)
    tmp_config = config.clone()
    tmp_config.merge_from_list(['federate.client_num', unified_client_num])
    translator = BaseDataTranslator(tmp_config, client_cfgs)
    syn_data_dict = translator(syn_split_datasets)
    logger.info(f"Successfully created federated SYN dataset with {len(syn_data_dict) - 1} clients")

    data_dicts = [mnist_data_dict, usps_data_dict, svhn_data_dict, syn_data_dict]
    # Merge all data dictionaries
    merged_data_dict = _merge_data_dictionaries(config, data_dicts)

    # Perform clustering analysis
    _perform_clustering_analysis(merged_data_dict, data_root)

    return merged_data_dict, config

def call_digits_data(config, client_cfgs=None):
    if config.data.type == "digits":
        return load_digits_data(config, client_cfgs)

from federatedscope.register import register_data
register_data("digits", call_digits_data)
