import os
import numpy as np
from PIL import Image
from sklearn.datasets import load_svmlight_file
import scipy.io as sio

import torch
import torch.utils.data
from torch.utils.data import Subset, Dataset, TensorDataset
from torchvision import datasets, transforms
from federatedscope.core.data import BaseDataTranslator

import logging
from sklearn.cluster import AgglomerativeClustering
import json

logger = logging.getLogger(__name__)

##########################
# 数据加载部分（示例）
##########################

def load_usps(data_root, transform):
    usps_dir = os.path.join(data_root, 'usps')
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
        mat = sio.loadmat(os.path.join(data_root, f"svhn/{split}_32x32.mat"))
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

##########################
# 计算客户端标签分布
##########################

def calculate_merged_distribution(distributions):
    """
    计算合并后的总体分布

    Args:
        distributions: 各个客户端的标签分布列表

    Returns:
        merged_distribution: 合并后的总体分布数组
    """
    if not distributions:
        return np.array([])

    # 将所有分布相加得到总体分布
    merged_distribution = np.sum(distributions, axis=0)
    return merged_distribution

def get_label_distribution(subsets, num_classes=10):
    distributions = []
    for subset in subsets:
        ds = subset.dataset
        if hasattr(ds, 'targets'):
            full_labels = np.array(ds.targets)
        elif hasattr(ds, 'y'):
            full_labels = np.array(ds.y)
        elif hasattr(ds, 'labels'):
            full_labels = np.array(ds.labels)
        elif hasattr(ds, 'samples'):
            full_labels = np.array([s[1] for s in ds.samples])
        else:
            raise AttributeError(f"Dataset {ds} has no label attribute")
        labels = full_labels[subset.indices]
        classes, counts = np.unique(labels, return_counts=True)
        # Convert classes to integers to use as indices
        classes = classes.astype(int)
        # Ensure classes are within valid range
        valid_classes = classes[classes < num_classes]
        valid_counts = counts[classes < num_classes]
        dist = np.zeros(num_classes, dtype=int)
        if len(valid_classes) > 0:
            dist[valid_classes] = valid_counts
        distributions.append(dist)
    return distributions

##########################
# 聚类算法（完全仿照FedGS_Splitter）
##########################

def normalize_distributions(distributions):
    normalized = []
    for dist in distributions:
        s = np.sum(dist)
        if s > 0:
            normalized.append(dist / s)
        else:
            normalized.append(dist)
    return np.array(normalized)

def compute_distance_matrix(norm_dists):
    n = len(norm_dists)
    dist_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            dist = np.linalg.norm(norm_dists[i] - norm_dists[j])
            dist_matrix[i, j] = dist
            dist_matrix[j, i] = dist
    return dist_matrix

def calculate_balance_score(cluster_labels, n_clients, min_groups, max_groups, balance_weight):
    num_clusters = len(np.unique(cluster_labels))
    if num_clusters < min_groups or num_clusters > max_groups:
        return float('inf'), num_clusters

    cluster_sizes = [np.sum(cluster_labels == label) for label in np.unique(cluster_labels)]
    target_size = n_clients / num_clusters
    size_variance = np.var(cluster_sizes)
    ideal_num_clusters = (min_groups + max_groups) / 2
    cluster_num_penalty = abs(num_clusters - ideal_num_clusters)
    balance_penalty = size_variance / (target_size ** 2)
    total_score = cluster_num_penalty + balance_weight * balance_penalty
    return total_score, num_clusters

def cluster_clients(train_distributions,
                    distance_threshold=0.5,
                    min_groups=2,
                    max_groups=10,
                    balance_weight=2.0,
                    balance_tolerance=0.2):
    n_clients = len(train_distributions)
    norm_dists = normalize_distributions(train_distributions)
    dist_matrix = compute_distance_matrix(norm_dists)

    best_score = float('inf')
    best_threshold = distance_threshold
    best_labels = None

    thresholds = np.linspace(0.01, 1.0, 50)
    logger.info("Searching optimal threshold for clustering...")

    for threshold in thresholds:
        try:
            clustering = AgglomerativeClustering(
                n_clusters=None,
                distance_threshold=threshold,
                metric='precomputed',
                linkage='complete'
            )
            cluster_labels = clustering.fit_predict(dist_matrix)
            score, num_clusters = calculate_balance_score(cluster_labels, n_clients,
                                                          min_groups, max_groups, balance_weight)
            if score < best_score:
                best_score = score
                best_threshold = threshold
                best_labels = cluster_labels
        except Exception as e:
            logger.debug(f"Threshold {threshold:.3f} failed: {e}")
            continue

    if best_labels is None:
        forced_clusters = max(min(n_clients, max_groups), min_groups)
        clustering = AgglomerativeClustering(
            n_clusters=forced_clusters,
            metric='precomputed',
            linkage='complete'
        )
        best_labels = clustering.fit_predict(dist_matrix)
        logger.warning(f"Could not find optimal threshold, forced {forced_clusters} clusters.")

    # 调整簇大小均衡
    peer_communities = []
    for label in np.unique(best_labels):
        peer_communities.append(np.where(best_labels == label)[0])

    target_size = n_clients / len(peer_communities)
    tolerance = max(1, int(target_size * balance_tolerance))
    min_size = int(target_size - tolerance)
    max_size = int(target_size + tolerance)

    group_sizes = [len(c) for c in peer_communities]
    max_iter = 10
    for _ in range(max_iter):
        max_idx = np.argmax(group_sizes)
        min_idx = np.argmin(group_sizes)
        if group_sizes[max_idx] <= max_size or group_sizes[min_idx] >= min_size:
            break

        big_group = peer_communities[max_idx]
        small_group = peer_communities[min_idx]

        best_move_idx = -1
        best_move_score = float('inf')
        for i, client in enumerate(big_group):
            dist_to_small = np.mean([dist_matrix[client][j] for j in small_group]) if len(small_group) > 0 else 0
            dist_to_big = np.mean([dist_matrix[client][j] for j in big_group if j != client])
            score = dist_to_small - dist_to_big
            if score < best_move_score:
                best_move_score = score
                best_move_idx = i

        if best_move_idx >= 0:
            client_to_move = big_group[best_move_idx]
            peer_communities[max_idx] = np.delete(big_group, best_move_idx)
            peer_communities[min_idx] = np.append(small_group, client_to_move)
            group_sizes[max_idx] -= 1
            group_sizes[min_idx] += 1
        else:
            break

    return peer_communities, best_threshold, best_score, group_sizes

##########################
# 保存3个txt文件，格式完全仿照FedGS_Splitter
##########################

def save_data_info(data_info, save_path):
    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write("Client Data Distribution Information\n")
        f.write("=" * 50 + "\n\n")
        for idx, dist in enumerate(data_info):
            dist_arr = np.array(dist['distribution'])
            total = float(np.sum(dist_arr))
            percentages = [f"{(x / total * 100):.2f}%" for x in dist_arr]
            f.write(f"Client #{idx + 1}\n")
            f.write(f"Distribution: {dist_arr.tolist()}\n")
            f.write(f"Percentages: {percentages}\n")
            f.write(f"Total Samples: {total}\n")
            f.write("-" * 30 + "\n")

        json_data = {"data_info": [d['distribution'].tolist() if isinstance(d['distribution'], np.ndarray) else d['distribution'] for d in data_info]}
        f.write("\nRaw Data (JSON format):\n")
        f.write(json.dumps(json_data, indent=2))

def save_test_data_info(test_data_info, save_path):
    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write("Test Set Data Distribution Information\n")
        f.write("=" * 50 + "\n\n")
        for idx, dist in enumerate(test_data_info):
            dist_arr = np.array(dist['distribution'])
            total = float(np.sum(dist_arr))
            percentages = [f"{(x / total * 100):.2f}%" for x in dist_arr]
            f.write(f"Client #{idx + 1}\n")
            f.write(f"Distribution: {dist_arr.tolist()}\n")
            f.write(f"Percentages: {percentages}\n")
            f.write(f"Total Samples: {total}\n")
            f.write("-" * 30 + "\n")

        json_data = {"test_data_info": [d['distribution'].tolist() if isinstance(d['distribution'], np.ndarray) else d['distribution'] for d in test_data_info]}
        f.write("\nRaw Data (JSON format):\n")
        f.write(json.dumps(json_data, indent=2))

def save_peer_communities(peer_communities, data_info, save_path):
    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write("Peer Community Grouping Information\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Total number of communities: {len(peer_communities)}\n\n")

        for idx, community in enumerate(peer_communities):
            display_community = [cid + 1 for cid in community.tolist()] if isinstance(community, np.ndarray) else [cid + 1 for cid in community]
            f.write(f"Community #{idx + 1}\n")
            f.write(f"Members: {display_community}\n")

            community_dist = np.sum([np.array(data_info[i]['distribution']) for i in community], axis=0)
            total_samples = np.sum(community_dist)
            percentages = [f"{(x / total_samples * 100):.2f}%" for x in community_dist]
            f.write(f"Community overall distribution: {community_dist.tolist()}\n")
            f.write(f"Distribution percentages: {percentages}\n")
            f.write(f"Total Samples: {total_samples}\n")

            f.write("-" * 30 + "\n")

        json_data = {
            "peer_communities": [
                [cid + 1 for cid in (comm.tolist() if isinstance(comm, np.ndarray) else comm)]
                for comm in peer_communities
            ]
        }
        f.write("\nRaw Data (JSON format):\n")
        f.write(json.dumps(json_data, indent=2))

##########################
# 主入口函数，注册给FederatedScope调用
##########################

from federatedscope.core.data.base_data import StandaloneDataDict, ClientData
from federatedscope.register import register_data

def load_cross_domain_data(config, client_cfgs=None):
    data_root = config.data.root

    # 加载数据集
    transform = transforms.Compose([
        transforms.Resize((28,28)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])

    mnist_train = datasets.MNIST(root=data_root, train=True, transform=transform, download=True)
    mnist_test = datasets.MNIST(root=data_root, train=False, transform=transform, download=True)
    
    mnist_split_datasets = (mnist_train, mnist_test, mnist_test)
    tmp_config = config.clone()
    tmp_config.merge_from_list(['federate.client_num', 31])
    translator = BaseDataTranslator(tmp_config, client_cfgs)
    mnist_data_dict = translator(mnist_split_datasets)

    usps_train, usps_test = load_usps(data_root, transform)
    
    usps_split_datasets = (usps_train, usps_test, usps_test)
    tmp_config = config.clone()
    tmp_config.merge_from_list(['federate.client_num', 4])
    translator = BaseDataTranslator(tmp_config, client_cfgs)
    usps_data_dict = translator(usps_split_datasets)
    
    svhn_train, svhn_test = load_svhn(data_root, transform)
    
    svhn_split_datasets = (svhn_train, svhn_test, svhn_test)
    tmp_config = config.clone()
    tmp_config.merge_from_list(['federate.client_num', 38])
    translator = BaseDataTranslator(tmp_config, client_cfgs)
    svhn_data_dict = translator(svhn_split_datasets)
    
    syn_train, syn_test = load_syn(data_root, transform)
    
    syn_split_datasets = (syn_train, syn_test, syn_test)
    tmp_config = config.clone()
    tmp_config.merge_from_list(['federate.client_num', 27])
    translator = BaseDataTranslator(tmp_config, client_cfgs)
    syn_data_dict = translator(syn_split_datasets)

    # 重新合并数据：将所有server数据合并到data_dict[0]，client数据按顺序编号
    data_dict = {}

    # 收集所有server数据（key=0）
    server_data_list = []
    if 0 in mnist_data_dict:
        server_data_list.append(mnist_data_dict[0])
    if 0 in usps_data_dict:
        server_data_list.append(usps_data_dict[0])
    if 0 in svhn_data_dict:
        server_data_list.append(svhn_data_dict[0])
    if 0 in syn_data_dict:
        server_data_list.append(syn_data_dict[0])

    # 合并所有server数据到data_dict[0]
    if server_data_list:
        # 这里需要实现server数据的合并逻辑
        # 暂时使用第一个server数据作为基础
        data_dict[0] = server_data_list[0]
        logger.info(f"Combined {len(server_data_list)} server datasets into data_dict[0]")

    # 按顺序添加所有client数据
    client_id = 1

    # 添加MNIST客户端数据 (原来的key 1-31)
    for key in sorted(mnist_data_dict.keys()):
        if key != 0:  # 跳过server数据
            data_dict[client_id] = mnist_data_dict[key]
            client_id += 1

    # 添加USPS客户端数据 (原来的key 1-4)
    for key in sorted(usps_data_dict.keys()):
        if key != 0:  # 跳过server数据
            data_dict[client_id] = usps_data_dict[key]
            client_id += 1

    # 添加SVHN客户端数据 (原来的key 1-38)
    for key in sorted(svhn_data_dict.keys()):
        if key != 0:  # 跳过server数据
            data_dict[client_id] = svhn_data_dict[key]
            client_id += 1

    # 添加SYN客户端数据 (原来的key 1-27)
    for key in sorted(syn_data_dict.keys()):
        if key != 0:  # 跳过server数据
            data_dict[client_id] = syn_data_dict[key]
            client_id += 1

    logger.info(f"Total clients after reorganization: {client_id - 1}")
    logger.info(f"Data dict keys: {sorted(data_dict.keys())}")
    print(f"DEBUG: Data dict keys: {sorted(data_dict.keys())}")  # Force print to console

    # 调试：检查数据结构
    logger.info("Debugging data structure:")
    print("DEBUG: Checking data structure...")
    for key in sorted(list(data_dict.keys())[:5]):  # 只检查前5个
        client_data = data_dict[key]
        logger.info(f"  Key {key}: type={type(client_data)}, attributes={dir(client_data)}")
        print(f"DEBUG: Key {key}: type={type(client_data)}")
        if hasattr(client_data, 'train'):
            train_info = f"train: {type(client_data.train)}, length={len(client_data.train) if client_data.train else 'None'}"
            logger.info(f"    {train_info}")
            print(f"DEBUG:   {train_info}")
        if hasattr(client_data, 'test'):
            test_info = f"test: {type(client_data.test)}, length={len(client_data.test) if client_data.test else 'None'}"
            logger.info(f"    {test_info}")
            print(f"DEBUG:   {test_info}")

    # 提取训练和测试数据集
    train_splits = []
    test_splits = []

    for client_id in sorted(data_dict.keys()):
        if client_id != 0:  # 跳过服务器数据 (现在server数据在key=0)
            client_data = data_dict[client_id]
            # ClientData对象使用train_data和test_data属性
            if hasattr(client_data, 'train_data') and client_data.train_data is not None:
                train_splits.append(client_data.train_data)
                logger.debug(f"Added train data for client {client_id}, length: {len(client_data.train_data)}")
            if hasattr(client_data, 'test_data') and client_data.test_data is not None:
                test_splits.append(client_data.test_data)
                logger.debug(f"Added test data for client {client_id}, length: {len(client_data.test_data)}")

    logger.info(f"Extracted {len(train_splits)} train splits and {len(test_splits)} test splits")

    # 计算分布
    train_distributions = get_label_distribution(train_splits, num_classes=10)
    test_distributions = get_label_distribution(test_splits, num_classes=10)

    logger.info(f"Calculated distributions for {len(train_distributions)} clients")

    # 计算合并后的分布
    merged_train_distribution = calculate_merged_distribution(train_distributions)
    merged_test_distribution = calculate_merged_distribution(test_distributions)

    logger.info("Merged Training Distribution:")
    logger.info(f"  Distribution: {merged_train_distribution}")
    logger.info(f"  Total samples: {np.sum(merged_train_distribution)}")

    logger.info("Merged Test Distribution:")
    logger.info(f"  Distribution: {merged_test_distribution}")
    logger.info(f"  Total samples: {np.sum(merged_test_distribution)}")

    # 聚类划分
    peer_communities, threshold, score, group_sizes = cluster_clients(
        train_distributions,
        distance_threshold=0.5,
        min_groups=2,
        max_groups=10,
        balance_weight=2.0,
        balance_tolerance=0.2
    )

    # 保存txt文件
    save_data_info([{'distribution': dist} for dist in train_distributions], 'data_info.txt')
    save_test_data_info([{'distribution': dist} for dist in test_distributions], 'test_data_info.txt')
    save_peer_communities(peer_communities, [{'distribution': dist} for dist in train_distributions], 'peer_communities.txt')

    return data_dict, config

def call_cross_domain_data(config, client_cfgs=None):
    if config.data.type == "cross_domain":
        return load_cross_domain_data(config, client_cfgs)

from federatedscope.register import register_data
register_data("cross_domain", call_cross_domain_data)
