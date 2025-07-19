import os
import numpy as np
from PIL import Image
from sklearn.datasets import load_svmlight_file
import scipy.io as sio

import torch
import torch.utils.data
from torch.utils.data import Subset, Dataset, TensorDataset
from torchvision import datasets, transforms

import logging
from sklearn.cluster import AgglomerativeClustering
import json
from federatedscope.core.data import ClientData, StandaloneDataDict

logger = logging.getLogger(__name__)
import glob

from federatedscope.core.data import BaseDataTranslator
from federatedscope.register import register_data



##########################
# 使用FederatedScope的Dirichlet划分
##########################

from federatedscope.core.splitters.utils import dirichlet_distribution_noniid_slice

def dirichlet_split(dataset, num_clients, alpha=0.5, min_size=1):
    """
    使用FederatedScope的dirichlet_distribution_noniid_slice进行Dirichlet划分

    Args:
        dataset: 数据集对象
        num_clients: 客户端数量
        alpha: Dirichlet分布参数，越小越不均匀
        min_size: 每个客户端最少样本数

    Returns:
        client_indices: 每个客户端的样本索引列表
    """
    # 提取标签
    if hasattr(dataset, 'targets'):
        labels = np.array(dataset.targets)
    elif hasattr(dataset, 'y'):
        labels = np.array(dataset.y)
    elif hasattr(dataset, 'labels'):
        labels = np.array(dataset.labels)
    elif hasattr(dataset, 'samples'):
        labels = np.array([s[1] for s in dataset.samples])
    else:
        raise AttributeError("Cannot find labels in dataset")

    # 使用FederatedScope的dirichlet_distribution_noniid_slice函数
    logger.info(f"Using FederatedScope's dirichlet_distribution_noniid_slice with alpha={alpha}, min_size={min_size}")
    client_indices = dirichlet_distribution_noniid_slice(
        labels, num_clients, alpha, min_size=min_size
    )

    logger.info(f"Successfully created {len(client_indices)} client partitions")
    return client_indices

##########################
# 计算客户端标签分布
##########################

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

class UTHarDataset(Dataset):
    def __init__(self, data: np.array, label: np.array):
        self.data = data
        self.targets = label

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index, :, :, :], int(self.targets[index])

    def __iter__(self):
        """Make the dataset iterable for model builder compatibility"""
        for i in range(len(self)):
            yield self[i]

def load_dataset(config, client_cfgs=None):
    data_dir = os.path.join(config.data.root, 'ut_har', 'data')
    label_dir = os.path.join(config.data.root, 'ut_har', 'label')
    
    data_list = glob.glob(os.path.join(data_dir, '*.csv'))
    label_list = glob.glob(os.path.join(label_dir, '*.csv'))
    
    ut_har_data = {}
    
    for data_dir in data_list:
        data_name = os.path.splitext(os.path.basename(data_dir))[0]
        with open(data_dir, 'rb') as f:
            data = np.load(f)
            data = data.reshape(len(data), 1, 250, 90)
            data_norm = (data - np.min(data)) / (np.max(data) - np.min(data))
        ut_har_data[data_name] = torch.Tensor(data_norm)

    
    for label_dir in label_list:
        label_name = os.path.splitext(os.path.basename(label_dir))[0]
        with open(label_dir, 'rb') as f:
            label = np.load(f)
        ut_har_data[label_name] = torch.Tensor(label)
        
    
    # 输出每个数据项的类型、dtype和形状
    for key, value in ut_har_data.items():
        print(f"{key} 的类型: {type(value)}, dtype: {value.dtype}, shape: {value.shape}")
        
    train_dataset = UTHarDataset(ut_har_data['X_train'], ut_har_data['y_train'])
    val_dataset = UTHarDataset(ut_har_data['X_val'], ut_har_data['y_val'])
    test_dataset = UTHarDataset(ut_har_data['X_test'], ut_har_data['y_test'])
    
    alpha = 0.5
    if hasattr(config.data, 'splitter_args') and config.data.splitter_args:
        alpha = config.data.splitter_args[0].get('alpha', 0.5)
        
    # 按客户端数划分
    train_idxs = dirichlet_split(train_dataset, 25, alpha)
    val_idxs = dirichlet_split(val_dataset, 25, alpha)
    test_idxs = dirichlet_split(test_dataset, 25, alpha)
    
    train_splits = []
    val_splits = []
    test_splits = []

    for idxs in train_idxs:
        train_splits.append(Subset(train_dataset, idxs))
    for idxs in val_idxs:
        val_splits.append(Subset(val_dataset, idxs))
    for idxs in test_idxs:
        test_splits.append(Subset(test_dataset, idxs))
        
    # 计算分布
    train_distributions = get_label_distribution(train_splits, num_classes=7)
    test_distributions = get_label_distribution(test_splits, num_classes=7)
    
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
    
    # Build data dict with `ClientData`, key `0` for server.
    data_dict = {
		0: ClientData(config, train=train_dataset, val=val_dataset, test=test_dataset)
	}
    for client_id in range(1, len(train_splits) + 1):
        client_cfg = config
        data_dict[client_id] = ClientData(client_cfg,
											train=train_splits[client_id - 1],
											val=val_splits[client_id - 1],
											test=test_splits[client_id - 1])
        
    data_dict_1 = StandaloneDataDict(data_dict, config)

    # split_datasets = (train_dataset, val_dataset, test_dataset)
    # split_datasets = (train_dataset, test_dataset, test_dataset)
    # tmp_config = config.clone()
    # tmp_config.merge_from_list(['federate.client_num', 25])
    # translator = BaseDataTranslator(tmp_config, client_cfgs)
    # data_dict_2 = translator(split_datasets)

    # 使用第二个data_dict作为最终结果
    data_dict = data_dict_1

    return data_dict, config

def call_cross_domain_data(config, client_cfgs=None):
    if config.data.type == "ut_har_test":
        return load_dataset(config, client_cfgs)

from federatedscope.register import register_data
register_data("ut_har_test", call_cross_domain_data)
