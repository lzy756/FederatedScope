import os
import json
import random
import logging

import numpy as np
import torch
from torch.utils.data import Subset, TensorDataset
from torchvision import datasets, transforms
from scipy.stats import entropy
from sklearn.cluster import DBSCAN

from federatedscope.core.data.base_data import StandaloneDataDict, ClientData
from federatedscope.register import register_data

logger = logging.getLogger(__name__)


##########################
# Dirichlet 划分客户端
##########################
def dirichlet_split(dataset, num_clients, alpha=0.5):
    labels = np.array([s[1] for s in dataset])
    num_classes = labels.max() + 1
    client_indices = [[] for _ in range(num_clients)]
    for k in range(num_classes):
        idx_k = np.where(labels == k)[0]
        np.random.shuffle(idx_k)
        props = np.random.dirichlet([alpha] * num_clients)
        counts = (props * len(idx_k)).astype(int)
        counts[-1] = len(idx_k) - counts[:-1].sum()
        start = 0
        for i, c in enumerate(counts):
            client_indices[i].extend(idx_k[start:start + c])
            start += c
    for lst in client_indices:
        random.shuffle(lst)
    return client_indices


##########################
# 计算标签分布
##########################
def get_label_distribution(subsets, num_classes=10):
    dists = []
    for subset in subsets:
        # 强制转 int，防止出现 float 标签
        labels = np.array([int(s[1]) for s in subset], dtype=int)
        dist = np.bincount(labels, minlength=num_classes)
        dists.append(dist)
    return dists


##########################
# JS 散度相关
##########################
def js_divergence(p, q):
    p = np.asarray(p, dtype=np.float64) + 1e-10
    q = np.asarray(q, dtype=np.float64) + 1e-10
    p /= p.sum()
    q /= q.sum()
    m = 0.5 * (p + q)
    return 0.5 * (entropy(p, m) + entropy(q, m))


def compute_jsd_matrix(norm_dists):
    n = len(norm_dists)
    mat = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            d = np.sqrt(js_divergence(norm_dists[i], norm_dists[j]))
            mat[i, j] = mat[j, i] = d
    return mat


def normalize_distributions(dists):
    out = []
    for d in dists:
        s = d.sum()
        out.append(d / s if s > 0 else d)
    return np.vstack(out)


##########################
# 自适应聚类
##########################
def adaptive_cluster_clients(
    train_dists,
    target_avg_size=4,
    min_threshold=0.01,
    max_threshold=1.0,
    max_iter=15,
    tolerance=1,
):
    n = len(train_dists)
    target_clusters = max(1, n // target_avg_size)
    norm_d = normalize_distributions(train_dists)
    jsd = compute_jsd_matrix(norm_d)

    low, high = min_threshold, max_threshold
    best_labels = None

    for _ in range(max_iter):
        mid = (low + high) / 2
        labels = DBSCAN(eps=mid, min_samples=1, metric="precomputed").fit_predict(jsd)
        sizes = [np.sum(labels == lb) for lb in np.unique(labels)]
        avg = np.mean(sizes)
        if abs(avg - target_avg_size) <= tolerance:
            best_labels = labels
            break
        if avg > target_avg_size:
            high = mid
        else:
            low = mid
        best_labels = labels

    communities = [
        list(np.where(best_labels == lb)[0]) for lb in np.unique(best_labels)
    ]
    return communities


##########################
# 保存 / 读取 聚类结果
##########################
def save_cluster_results(filename, train_dists, test_dists, communities):
    """
    保存聚类结果到 JSON 文件，并确保所有的 numpy.int64 都转换为 Python int
    :param filename: 保存路径
    :param train_dists: list of 1d numpy arrays or lists
    :param test_dists: list of 1d numpy arrays or lists
    :param communities: list of lists (可能包含 numpy.int64)
    """
    # 先把 train/test 分布全部转换为纯 Python int 的嵌套列表
    train_list = []
    for dist in train_dists:
        # 如果是 ndarray，就用 .tolist()，否则假定是 list，但可能内部还是 numpy.int64
        seq = dist.tolist() if isinstance(dist, np.ndarray) else dist
        train_list.append([int(x) for x in seq])

    test_list = []
    for dist in test_dists:
        seq = dist.tolist() if isinstance(dist, np.ndarray) else dist
        test_list.append([int(x) for x in seq])

    # 同理处理 communities（list of lists）
    comm_list = []
    for community in communities:
        # 如果 community 本身是 ndarray，就先 tolist()
        seq = community.tolist() if isinstance(community, np.ndarray) else community
        comm_list.append([int(x) for x in seq])

    data = {
        "metadata": {
            "num_clients": int(len(train_list)),
            "num_classes": int(len(train_list[0]) if train_list else 0),
            "num_communities": int(len(comm_list)),
        },
        "train_distributions": train_list,
        "test_distributions": test_list,
        "peer_communities": comm_list,
    }

    with open(filename, "w") as f:
        json.dump(data, f, indent=2)

    logger.info(f"保存成功！文件: {filename}")


def load_cluster_results(path):
    with open(path, "r") as f:
        data = json.load(f)
    td = [np.array(d) for d in data["train_distributions"]]
    vd = [np.array(d) for d in data["test_distributions"]]
    pc = data["peer_communities"]
    return td, vd, pc


##########################
# 采样函数
##########################
def sample_data(ds, fraction=0.4):
    n = len(ds)
    idx = list(range(n))
    random.shuffle(idx)
    m = int(n * fraction)
    return Subset(ds, idx[:m])


##########################
# 主加载函数
##########################
def load_office_caltech10(config, client_cfgs=None):
    data_root = config.data.root
    alpha = config.data.splitter_args[0].get("alpha", 0.5)
    total_clients = config.federate.get("client_num", 100)

    # train-transform
    train_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3),
    ])

    # 四个域
    domains = ["amazon", "caltech", "dslr", "webcam"]
    train_ds = []
    test_imgs = []
    test_labels = []

    for dm in domains:
        root = os.path.join(data_root, dm)
        tr_dir = os.path.join(root, "train")
        te_dir = os.path.join(root, "test")
        if not os.path.isdir(tr_dir):
            tr_dir = root
        if not os.path.isdir(te_dir):
            te_dir = root

        # train dataset
        ds_tr = datasets.ImageFolder(tr_dir, transform=train_tf)
        train_ds.append(sample_data(ds_tr))

        # raw test images
        ds_te = datasets.ImageFolder(te_dir, transform=None)
        for img, lb in ds_te:
            # img is PIL
            test_imgs.append(train_tf(img))
            test_labels.append(lb)

    # concat test
    test_imgs = torch.stack(test_imgs)
    test_labels = torch.tensor(test_labels)
    total_test = len(test_imgs)

    # split test to clients
    indices = list(range(total_test))
    random.shuffle(indices)
    base = total_test // total_clients
    rem = total_test % total_clients
    test_splits = []
    st = 0
    for i in range(total_clients):
        en = st + base + (1 if i < rem else 0)
        idx = indices[st:en]
        ds = TensorDataset(test_imgs, test_labels)
        test_splits.append(Subset(ds, idx))
        st = en

    # flatten train splits
    train_splits = []
    for ds in train_ds:
        # how many clients for this domain?
        num = int(np.ceil(len(ds)/ sum(len(x) for x in train_ds) * total_clients))
        if ds is train_ds[-1]:
            num = total_clients - sum(int(np.ceil(len(x)/ sum(len(y) for y in train_ds) * total_clients)) for x in train_ds[:-1])
        parts = dirichlet_split(ds, num, alpha)
        train_splits += [Subset(ds, p) for p in parts]

    # distributions
    train_dists = get_label_distribution(train_splits, num_classes=10)
    test_dists  = get_label_distribution(test_splits,  num_classes=10)

    # clustering
    communities = adaptive_cluster_clients(train_dists)
    logger.info(f"Peer communities: {communities}")
    save_cluster_results(
        os.path.join(data_root, "peer_communities.json"),
        train_dists, test_dists, communities
    )

    # pack into FederatedScope
    cd = {}
    for cid, (tr, te) in enumerate(zip(train_splits, test_splits), start=1):
        cd[cid] = ClientData(client_cfg=config, train=tr, val=None, test=te)
    fsdata = StandaloneDataDict(cd, config)
    return fsdata, config


def call_office_caltech10_data(config, client_cfgs=None):
    if config.data.type == "office_caltech10":
        return load_office_caltech10(config, client_cfgs)


if __name__ == "__main__":
    from federatedscope.core.configs.config import global_cfg
    global_cfg.data.root = os.path.expanduser("~/FederatedScope/data/office_caltech_10")
    global_cfg.data.type = "office_caltech10"
    data, cfg = load_office_caltech10(global_cfg)
else:
    register_data("office_caltech10", call_office_caltech10_data)
