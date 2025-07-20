import os
import numpy as np
import random
from PIL import Image
from sklearn.datasets import load_svmlight_file
import scipy.io as sio

import torch
import torch.utils
from torch.utils.data import Subset, Dataset, TensorDataset
import torch.utils.data
from torchvision import datasets, transforms

import logging

# from sklearn.cluster import AgglomerativeClustering
import json

logger = logging.getLogger(__name__)

##########################
# 数据加载部分（示例）
##########################


def load_usps(data_root, transform):
    usps_dir = os.path.join(data_root, "USPS")
    X_tr_sp, y_tr = load_svmlight_file(os.path.join(usps_dir, "usps"), n_features=256)
    X_te_sp, y_te = load_svmlight_file(os.path.join(usps_dir, "usps.t"), n_features=256)
    X_tr = X_tr_sp.toarray().reshape(-1, 16, 16).astype("uint8")
    X_te = X_te_sp.toarray().reshape(-1, 16, 16).astype("uint8")
    y_tr = y_tr.astype(int) - 1
    y_te = y_te.astype(int) - 1

    class USPSDataset(TensorDataset):
        def __init__(self, X, y, transform):
            self.X, self.y, self.transform = X, y, transform

        def __len__(self):
            return len(self.y)

        def __getitem__(self, idx):
            img = Image.fromarray(self.X[idx], mode="L")
            img = self.transform(img)
            return img, int(self.y[idx])

    return USPSDataset(X_tr, y_tr, transform), USPSDataset(X_te, y_te, transform)


def load_svhn(data_root, transform):
    def load_mat_svhn(split):
        mat = sio.loadmat(os.path.join(data_root, f"SVHN/{split}_32x32.mat"))
        X = np.transpose(mat["X"], (3, 0, 1, 2))
        y = mat["y"].reshape(-1).astype(int)
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

    return load_mat_svhn("train"), load_mat_svhn("test")


def load_syn(data_root, transform):
    from torchvision.datasets import ImageFolder

    syn_root = os.path.join(data_root, "SYN")
    train_ds = ImageFolder(
        root=os.path.join(syn_root, "imgs_train"), transform=transform
    )
    test_ds = ImageFolder(
        root=os.path.join(syn_root, "imgs_valid"), transform=transform
    )
    return train_ds, test_ds


##########################
# Dirichlet划分客户端
##########################


def dirichlet_split(dataset, num_clients, alpha=0.5):
    if hasattr(dataset, "targets"):
        labels = np.array(dataset.targets)
    elif hasattr(dataset, "y"):
        labels = np.array(dataset.y)
    elif hasattr(dataset, "labels"):
        labels = np.array(dataset.labels)
    elif hasattr(dataset, "samples"):
        labels = np.array([s[1] for s in dataset.samples])
    else:
        raise AttributeError("Cannot find labels in dataset")

    num_classes = labels.max() + 1
    client_indices = [[] for _ in range(num_clients)]

    for k in range(num_classes):
        idx_k = np.where(labels == k)[0]
        np.random.shuffle(idx_k)
        proportions = np.random.dirichlet([alpha] * num_clients)
        counts = (proportions * len(idx_k)).astype(int)
        counts[-1] = len(idx_k) - counts[:-1].sum()
        start = 0
        for client_id, c in enumerate(counts):
            client_indices[client_id].extend(idx_k[start : start + c])
            start += c

    for c in client_indices:
        random.shuffle(c)
    return client_indices


##########################
# 计算客户端标签分布
##########################


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


##########################
# 聚类算法（完全仿照FedGS_Splitter）
##########################


def normalize_distributions(distributions: list):
    normalized = []
    for dist in distributions:
        s = np.sum(dist)
        if s > 0:
            normalized.append(dist / s)
        else:
            normalized.append(dist)
    return np.array(normalized)


from sklearn.cluster import DBSCAN
from scipy.stats import entropy
import numpy as np


import logging

logger = logging.getLogger(__name__)


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


##########################
# 保存3个txt文件，格式完全仿照FedGS_Splitter
##########################
def load_cluster_results(filename):
    """从JSON文件加载聚类结果

    返回: (train_dists, test_dists, communities)
    """
    with open(filename, "r") as f:
        data = json.load(f)

    # 转换回numpy数组
    train_dists = [np.array(dist) for dist in data["train_distributions"]]
    test_dists = [np.array(dist) for dist in data["test_distributions"]]
    # peer_communities = [np.array(community) for community in data["peer_communities"]]

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


##########################
# 主入口函数，注册给FederatedScope调用
##########################

from federatedscope.core.data.base_data import StandaloneDataDict, ClientData
from federatedscope.register import register_data


def load_cross_domain_data(config, client_cfgs=None):
    data_root = config.data.root

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

    mnist_train = datasets.MNIST(
        root=data_root, train=True, transform=transform, download=True
    )
    mnist_test = datasets.MNIST(
        root=data_root, train=False, transform=transform, download=True
    )

    usps_train, usps_test = load_usps(data_root, transform)
    svhn_train, svhn_test = load_svhn(data_root, transform)
    syn_train, syn_test = load_syn(data_root, transform)
    logger.info(
        f"Loaded datasets: MNIST, USPS, SVHN, SYN with {len(mnist_train)}, {len(usps_train)}, {len(svhn_train)}, {len(syn_train)} "
        f"train samples and {len(mnist_test)}, {len(usps_test)}, {len(svhn_test)}, {len(syn_test)} test samples each."
    )
    mnist_test_dataloader = torch.utils.data.DataLoader(
        mnist_test, batch_size=1000, shuffle=False, num_workers=4
    )
    usps_test_dataloader = torch.utils.data.DataLoader(
        usps_test, batch_size=1000, shuffle=False, num_workers=4
    )
    svhn_test_dataloader = torch.utils.data.DataLoader(
        svhn_test, batch_size=1000, shuffle=False, num_workers=4
    )
    syn_test_dataloader = torch.utils.data.DataLoader(
        syn_test, batch_size=1000, shuffle=False, num_workers=4
    )

    # 所有测试集的图片
    all_test_images = []
    all_test_labels = []
    for dataloader in [
        mnist_test_dataloader,
        usps_test_dataloader,
        svhn_test_dataloader,
        syn_test_dataloader,
    ]:
        for images, labels in dataloader:
            all_test_images.append(images)
            all_test_labels.extend(labels.numpy().tolist())

    all_test_images = torch.cat(all_test_images, dim=0)
    all_test_datasets = torch.utils.data.TensorDataset(
        all_test_images, torch.tensor(all_test_labels)
    )

    logger.info(f"Total test samples: {len(all_test_datasets)}")

    # 打乱测试集
    total_test_samples = len(all_test_datasets)
    all_test_indices = list(range(total_test_samples))
    random.shuffle(all_test_indices)
    # 计算测试集总样本数
    # total_test_samples = sum(len(ds) for ds in all_test_datasets)
    total_clients = 100
    per_client_test_size = total_test_samples // total_clients
    remainder = total_test_samples % total_clients
    test_splits = []
    start = 0
    for i in range(total_clients):
        end = start + per_client_test_size + (1 if i < remainder else 0)
        test_splits.append(Subset(all_test_datasets, all_test_indices[start:end]))
        start = end

    # 训练集按照不同的域名进行划分
    mnist_train_idxs = dirichlet_split(mnist_train, 31, alpha=0.1)
    usps_train_idxs = dirichlet_split(usps_train, 4, alpha=0.1)
    svhn_train_idxs = dirichlet_split(svhn_train, 38, alpha=0.1)
    syn_train_idxs = dirichlet_split(syn_train, 27, alpha=0.1)
    train_splits = []

    train_splits.extend([Subset(mnist_train, idxs) for idxs in mnist_train_idxs])
    train_splits.extend([Subset(usps_train, idxs) for idxs in usps_train_idxs])
    train_splits.extend([Subset(svhn_train, idxs) for idxs in svhn_train_idxs])
    train_splits.extend([Subset(syn_train, idxs) for idxs in syn_train_idxs])

    # 计算分布
    train_distributions = get_label_distribution(train_splits, num_classes=10)
    test_distributions = get_label_distribution(test_splits, num_classes=10)

    # 聚类划分
    peer_communities = adaptive_cluster_clients(train_distributions)

    logger.info(f"Peer communities: {peer_communities}")
    # 保存聚类结果
    save_cluster_results(
        os.path.join(data_root, "peer_communities.json"),
        train_distributions,
        test_distributions,
        peer_communities,
    )

    # 构建 FederatedScope 格式数据
    client_data_dict = {}
    from federatedscope.core.data.base_data import ClientData, StandaloneDataDict

    for cid, (tr, te) in enumerate(zip(train_splits, test_splits), start=1):
        client_data_dict[cid] = ClientData(
            client_cfg=config, train=tr, val=None, test=te
        )
    fs_data = StandaloneDataDict(client_data_dict, config)
    return fs_data, config


def call_cross_domain_data(config, client_cfgs=None):
    if config.data.type == "cross_domain":
        return load_cross_domain_data(config, client_cfgs)


if __name__ == "__main__":
    from federatedscope.core.configs.config import global_cfg

    # 设置数据根目录
    global_cfg.data.root = "./data"
    # 设置数据类型
    global_cfg.data.type = "cross_domain"
    # 调用数据加载函数
    fs_data, config = load_cross_domain_data(global_cfg)
else:
    # 如果不是直接运行此脚本，则注册数据加载函数
    from federatedscope.register import register_data

    register_data("cross_domain", call_cross_domain_data)
