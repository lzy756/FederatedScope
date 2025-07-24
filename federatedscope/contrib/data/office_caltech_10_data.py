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
# Dirichlet 划分客户端，并确保至少一个样本
##########################
def dirichlet_split_with_minimum(dataset, num_clients, alpha=0.5):
    labels = np.array([s[1] for s in dataset])
    num_classes = labels.max() + 1
    # 先按 Dirichlet 拆分
    client_indices = [[] for _ in range(num_clients)]
    for k in range(num_classes):
        idx_k = np.where(labels == k)[0]
        np.random.shuffle(idx_k)
        props = np.random.dirichlet([alpha] * num_clients)
        counts = (props * len(idx_k)).astype(int)
        # 最后一个桶补齐
        counts[-1] = len(idx_k) - counts[:-1].sum()
        start = 0
        for i, c in enumerate(counts):
            client_indices[i].extend(idx_k[start:start + c])
            start += c
    # 确保每个桶至少 1 个样本
    # 如果有空桶，就从其他桶中 “借” 一个
    for i in range(num_clients):
        if len(client_indices[i]) == 0:
            # 找一个有多余的桶
            for j in range(num_clients):
                if len(client_indices[j]) > 1:
                    borrowed = client_indices[j].pop()
                    client_indices[i].append(borrowed)
                    break
    # 最后打乱每个桶内部的顺序
    for lst in client_indices:
        random.shuffle(lst)
    return client_indices

##########################
# 统计标签分布
##########################
def get_label_distribution(subsets, num_classes=10):
    dists = []
    for subset in subsets:
        labels = np.array([int(s[1]) for s in subset], dtype=int)
        dist = np.bincount(labels, minlength=num_classes)
        dists.append(dist.tolist())
    return dists

##########################
# JS 散度相关
##########################
def js_divergence(p, q):
    p = np.asarray(p, dtype=np.float64) + 1e-10
    q = np.asarray(q, dtype=np.float64) + 1e-10
    p /= p.sum(); q /= q.sum()
    m = 0.5 * (p + q)
    return 0.5 * (entropy(p,m) + entropy(q,m))

def compute_jsd_matrix(norm_dists):
    n = len(norm_dists)
    mat = np.zeros((n,n), dtype=np.float64)
    for i in range(n):
        for j in range(i+1, n):
            d = np.sqrt(js_divergence(norm_dists[i], norm_dists[j]))
            mat[i,j] = mat[j,i] = d
    return mat

def normalize_distributions(dists):
    out = []
    for d in dists:
        s = sum(d)
        out.append([x/s if s>0 else x for x in d])
    return np.array(out)

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
    target = max(1, n//target_avg_size)
    norm_d = normalize_distributions(train_dists)
    jsd = compute_jsd_matrix(norm_d)
    low, high = min_threshold, max_threshold
    best_labels = None

    for _ in range(max_iter):
        mid = (low + high)/2
        labels = DBSCAN(eps=mid, min_samples=1, metric="precomputed").fit_predict(jsd)
        sizes = [np.sum(labels==lb) for lb in np.unique(labels)]
        avg = np.mean(sizes)
        best_labels = labels
        if abs(avg - target_avg_size) <= tolerance:
            break
        if avg > target_avg_size:
            high = mid
        else:
            low = mid

    communities = [list(np.where(best_labels==lb)[0]) for lb in np.unique(best_labels)]
    return communities

##########################
# 保存 / 读取 聚类结果
##########################
def save_cluster_results(path, train_dists, test_dists, communities):
    # 将所有 numpy 转成纯 Python int/list
    train_list = [[int(x) for x in dist] for dist in train_dists]
    test_list  = [[int(x) for x in dist] for dist in test_dists]
    comm_list  = [[int(x) for x in c]    for c in communities]

    data = {
        "metadata": {
            "num_clients": len(train_list),
            "num_classes": len(train_list[0]) if train_list else 0,
            "num_communities": len(comm_list),
        },
        "train_distributions": train_list,
        "test_distributions":  test_list,
        "peer_communities":    comm_list,
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    logger.info(f"Saved cluster results to {path}")

def load_cluster_results(path):
    with open(path,"r") as f:
        data = json.load(f)
    td = [np.array(d) for d in data["train_distributions"]]
    vd = [np.array(d) for d in data["test_distributions"]]
    pc = data["peer_communities"]
    return td, vd, pc

##########################
# 采样 70% 训练 / 30% 测试 （按域内部）
##########################
def split_train_test(dataset, frac=0.7):
    n = len(dataset)
    idx = list(range(n))
    random.shuffle(idx)
    m = int(n * frac)
    return idx[:m], idx[m:]

##########################
# 主加载函数
##########################
def load_office_caltech10(config, client_cfgs=None):
    data_root     = os.path.expanduser(config.data.root)
    alpha         = config.data.splitter_args[0].get("alpha", 0.5)
    total_clients = config.federate.get("client_num", 100)

    # 图像预处理（训练/测试同用）
    tf = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3),
    ])

    domains = ["amazon","caltech","dslr","webcam"]

    # 1) 对每个域，按 70/30 划分训练池 / 测试池
    domain_train_pools = {}
    domain_test_pools  = {}
    for dm in domains:
        root_dir = os.path.join(data_root, dm)
        # 如果有 train/ test 子目录则优先，否则整个文件夹
        trd = os.path.join(root_dir,"train")
        ted = os.path.join(root_dir,"test")
        trd = trd if os.path.isdir(trd) else root_dir
        ted = ted if os.path.isdir(ted) else root_dir

        ds_raw  = datasets.ImageFolder(trd, transform=None)   # 先无 transform
        tr_idx, te_idx = split_train_test(ds_raw, frac=0.7)
        # 创建带 transform 的子集
        ds_tr = Subset(datasets.ImageFolder(trd, transform=tf), tr_idx)
        ds_te = Subset(datasets.ImageFolder(ted, transform=tf), te_idx)
        domain_train_pools[dm] = ds_tr
        domain_test_pools[dm]  = ds_te

    # 2) 计算每个域分配多少个客户端（按训练样本数比例）
    train_sizes = [len(domain_train_pools[dm]) for dm in domains]
    total_train = sum(train_sizes)
    clients_per_domain = [ int(np.ceil(sz/total_train*total_clients)) for sz in train_sizes ]
    # 修正最后一个域，确保总数一致
    clients_per_domain[-1] = total_clients - sum(clients_per_domain[:-1])

    # 3) 用 Dirichlet 划分训练池到每个域内的客户端，并保证至少 1 张
    train_splits = []
    client_domains = []
    for dm, c_num in zip(domains, clients_per_domain):
        pool = domain_train_pools[dm]
        idx_splits = dirichlet_split_with_minimum(pool, c_num, alpha)
        for idxs in idx_splits:
            train_splits.append(Subset(pool, idxs))
            client_domains.append(dm)

    # 4) 将域测试池平均划分给本域的各个客户端
    test_splits = []
    for dm, c_num in zip(domains, clients_per_domain):
        pool = domain_test_pools[dm]
        n = len(pool)
        indices = list(range(n))
        random.shuffle(indices)
        base = n // c_num
        rem  = n %  c_num
        st = 0
        for i in range(c_num):
            en = st + base + (1 if i<rem else 0)
            test_splits.append(Subset(pool, indices[st:en]))
            st = en

    # 5) 统计每个客户端的标签分布，并打印
    train_dists = get_label_distribution(train_splits, num_classes = 10)
    test_dists  = get_label_distribution(test_splits,  num_classes = 10)
    print("===== Client train/test distributions =====")
    for cid, (dom, td, vd) in enumerate(zip(client_domains, train_dists, test_dists), start=1):
        print(f"Client{cid:02d} | Domain={dom:8s} | TrainDist={td} | TestDist={vd}")

    # 6) 自适应聚类（不影响数据划分，只做记录/存储）
    communities = adaptive_cluster_clients(train_dists)
    save_cluster_results(
        os.path.join(data_root,"peer_communities.json"),
        train_dists, test_dists, communities
    )

    # 7) 打包成 FederatedScope 所需格式
    client_data = {}
    for cid, (tr, te) in enumerate(zip(train_splits, test_splits), start=1):
        client_data[cid] = ClientData(client_cfg=config, train=tr, val=None, test=te)

    fsdata = StandaloneDataDict(client_data, config)
    return fsdata, config

def call_office_caltech10_data(config, client_cfgs=None):
    if config.data.type == "office_caltech10":
        return load_office_caltech10(config, client_cfgs)

if __name__ == "__main__":
    from federatedscope.core.configs.config import global_cfg
    global_cfg.data.root = "~/FederatedScope/data/office_caltech_10"
    global_cfg.data.type = "office_caltech10"
    data, cfg = load_office_caltech10(global_cfg)
else:
    register_data("office_caltech10", call_office_caltech10_data)
