import numpy as np
import random
from sklearn.cluster import DBSCAN
from scipy.stats import entropy
import numpy as np
from torch.utils.data import Subset

import json
import logging

logger = logging.getLogger(__name__)


def dirichlet_split(dataset, num_clients, alpha=0.5):
    labels = np.array([s[1] for s in dataset])
    num_classes = labels.max() + 1
    client_indices = [[] for _ in range(num_clients)]

    for k in range(num_classes):
        idx_k = np.where(labels == k)[0]
        np.random.shuffle(idx_k)
        proportions = np.random.dirichlet([alpha] * num_clients)

        proportions = np.clip(proportions, 1e-8, None)
        proportions /= proportions.sum()

        counts = (proportions * len(idx_k)).round().astype(int)
        remainder = len(idx_k) - counts.sum()

        if remainder > 0:
            # 如果有剩余样本，均匀分配到前几个客户端
            counts[:remainder] += 1
        # counts[-1] = len(idx_k) - counts[:-1].sum()
        start = 0
        for client_id, c in enumerate(counts):
            client_indices[client_id].extend(idx_k[start : start + c])
            start += c
    total_samples = sum(len(c) for c in client_indices)
    logger.info(
        f"Total samples: {total_samples}, Dataset size: {len(dataset)}, Clients: {num_clients}"
    )
    assert total_samples == len(dataset), "Sample count mismatch after splitting"

    for c in client_indices:
        random.shuffle(c)
    return client_indices


def uniform_split(dataset, num_clients):
    """
    均匀划分数据集，确保每个客户端获得相同数量的样本，且类别分布均匀

    参数:
    dataset -- 要划分的数据集（包含特征和标签）
    num_clients -- 客户端数量

    返回:
    client_indices -- 包含每个客户端样本索引的列表
    """
    total_samples = len(dataset)

    # 计算每个客户端的样本数量（均匀分配）
    samples_per_client = total_samples // num_clients
    remainder = total_samples % num_clients  # 处理余数

    # 初始化客户端索引列表
    client_indices = [[] for _ in range(num_clients)]

    # 对每个类别进行统一分配
    all_indices = np.arange(total_samples)
    np.random.shuffle(all_indices)  # 随机打乱所有样本索引

    # 分配样本给每个客户端
    start = 0
    for i in range(num_clients):
        # 确定当前客户端的样本数量（处理余数）
        end = start + samples_per_client + (1 if i < remainder else 0)
        client_indices[i] = all_indices[start:end].tolist()
        start = end

        # 打乱当前客户端内部的样本顺序
        random.shuffle(client_indices[i])

    return client_indices


def normalize_distributions(distributions: list):
    normalized = []
    for dist in distributions:
        s = np.sum(dist)
        if s > 0:
            normalized.append(dist / s)
        else:
            normalized.append(dist)
    return np.array(normalized)


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

    参数:
        target_avg_size: 每个簇的目标平均大小
        min_threshold: JSD阈值下限
        max_threshold: JSD阈值上限
        max_iter: 最大调整迭代次数
        tolerance: 可接受的平均簇大小波动范围
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


def load_cluster_results(filename):
    """从JSON文件加载聚类结果

    返回: (train_dists, test_dists, communities)
    """
    with open(filename, "r") as f:
        data = json.load(f)

    # 转换回numpy数组
    train_dists = [np.array(dist) for dist in data["train_distributions"]]
    test_dists = [np.array(dist) for dist in data["test_distributions"]]

    return train_dists, test_dists, data["peer_communities"]


def save_cluster_results(filename, train_dists, test_dists, communities):
    """保存聚类结果到JSON文件

    :param filename: 保存路径 (e.g., 'cluster_results.json')
    :param train_dists: 训练分布列表 (num_clients x num_classes)
    :param test_dists: 测试分布列表 (num_clients x num_classes)
    :param communities: 社区结构 [[client_idx1, ...], ...]
    """
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

    logger.info(f"保存成功！文件: {filename}")


def sample_data(dataset, samples=2000):
    size = len(dataset)
    indices = list(range(size))
    random.shuffle(indices)
    sampled_indices = indices[:samples]
    return Subset(dataset, sampled_indices)


def get_label_distribution(subsets: list, num_classes=10):
    """
    计算每个子集的标签分布
    :param subsets: list of Subset objects
    :param num_classes: number of classes in the dataset
    :return: list of distributions for each subset
    """
    distributions = []
    for subset in subsets:
        labels = np.array([s[1] for s in subset])
        dist = np.zeros(num_classes, dtype=int)
        for label in labels:
            dist[label] += 1
        distributions.append(dist)
    return distributions
