import os
import json
import random
import logging
import time

from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import Subset, TensorDataset
from torchvision import datasets, transforms
from scipy.stats import entropy
from sklearn.cluster import DBSCAN

from federatedscope.core.data.base_data import StandaloneDataDict, ClientData
from federatedscope.register import register_data

logger = logging.getLogger(__name__)

logging.basicConfig(
    format="%(asctime)s %(levelname)s %(message)s",
    level=logging.INFO,
    datefmt="%H:%M:%S"
)
##########################
# Dirichlet 划分客户端，并确保至少一个样本
##########################
def dirichlet_split_with_minimum(dataset, num_clients, alpha=0.5):
    labels = np.array([s[1] for s in dataset])
    num_classes = labels.max() + 1
    client_indices = [[] for _ in range(num_clients)]
    # Dirichlet 拆分
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
    # 保证每个客户端至少一个样本
    for i in range(num_clients):
        if len(client_indices[i]) == 0:
            for j in range(num_clients):
                if len(client_indices[j]) > 1:
                    client_indices[i].append(client_indices[j].pop())
                    break
    # 打乱
    for lst in client_indices:
        random.shuffle(lst)
    return client_indices

##########################
# 统计标签分布
##########################
def get_label_distribution(subsets, num_classes=7):
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
    return 0.5 * (entropy(p, m) + entropy(q, m))

def compute_jsd_matrix(norm_dists):
    n = len(norm_dists)
    mat = np.zeros((n, n), dtype=np.float64)
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
    norm_d = normalize_distributions(train_dists)
    jsd = compute_jsd_matrix(norm_d)
    low, high = min_threshold, max_threshold
    best_labels = None

    for _ in range(max_iter):
        mid = (low + high) / 2
        labels = DBSCAN(eps=mid, min_samples=1, metric="precomputed")\
                   .fit_predict(jsd)
        sizes = [np.sum(labels == lb) for lb in np.unique(labels)]
        avg = np.mean(sizes) if sizes else 0
        best_labels = labels
        if abs(avg - target_avg_size) <= tolerance:
            break
        if avg > target_avg_size:
            high = mid
        else:
            low = mid

    communities = [list(np.where(best_labels == lb)[0])
                   for lb in np.unique(best_labels)]
    return communities

##########################
# 保存 / 读取 聚类结果
##########################
def save_cluster_results(path, train_dists, test_dists, communities):
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
# 按域内部 70% 训练 / 30% 测试
##########################
def split_train_test(dataset, frac=0.7):
    n = len(dataset)
    idx = list(range(n))
    random.shuffle(idx)
    m = int(n * frac)
    return idx[:m], idx[m:]

##########################
# 主加载函数 for PACS
##########################
def load_pacs(config, client_cfgs=None):
    data_root     = os.path.expanduser(config.data.root)
    alpha         = config.data.splitter_args[0].get("alpha", 0.5)
    total_clients = config.federate.get("client_num", 100)

    # 通用 transform
    tf = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3),
    ])

    domains = ["art_painting","cartoon","photo","sketch"]

    logging.info("=== 开始加载 PACS 数据 ===")
    t0 = time.time()

    # 1) 每个域 70/30 划分 train pool / test pool
    domain_train = {}
    domain_test  = {}
    for dm in domains:
        logging.info(f"[1/5] 域 {dm}：开始 70/30 划分")
        start = time.time()
        folder = os.path.join(data_root, dm)
        raw = datasets.ImageFolder(folder, transform=None)
        tr_idx, te_idx = split_train_test(raw, frac=0.7)
        # 构造带 transform 的 Subset
        ds_tr = Subset(datasets.ImageFolder(folder, transform=tf), tr_idx)
        ds_te = Subset(datasets.ImageFolder(folder, transform=tf), te_idx)
        domain_train[dm] = ds_tr
        domain_test[dm]  = ds_te
        logging.info(f"[1/5] 域 {dm}：完成，"
                     f"train={len(tr_idx)}, test={len(te_idx)}, "
                     f"耗时 {time.time()-start:.1f}s")

    # 2) 统计各域训练池大小，计算 clients_per_domain
    logging.info("[2/5] 计算各域客户端分配数量")
    train_sizes = [len(domain_train[dm]) for dm in domains]
    total_train = sum(train_sizes)
    clients_per_domain = [
        int(np.ceil(sz/total_train*total_clients)) for sz in train_sizes
    ]
    # 修正最后一域
    clients_per_domain[-1] = total_clients - sum(clients_per_domain[:-1])
    logging.info(f"[2/5] clients_per_domain = {dict(zip(domains, clients_per_domain))}")

    # 3) Dirichlet 划分训练集
    train_splits = []
    client_domains = []
    logging.info("[3/5] Dirichlet 划分训练集到客户端")
    t3 = time.time()
    for dm, c_num in zip(domains, clients_per_domain):
        logging.info(f"  域 {dm} → {c_num} 个客户端")
        start = time.time()
        idx_splits = dirichlet_split_with_minimum(domain_train[dm], c_num, alpha)
        for idxs in idx_splits:
            train_splits.append(Subset(domain_train[dm], idxs))
            client_domains.append(dm)
        logging.info(
            f"  域 {dm}：生成 {len(idx_splits)} 个 split，耗时 {time.time()-start:.1f}s"
        )
    logging.info(f"[3/5] 完成全部训练集划分，总耗时 {time.time()-t3:.1f}s")

    # 4) 将各域测试池平均划分给本域客户端
    test_splits = []
    logging.info("[4/5] 划分测试集到客户端（按域内部平均）")
    t4 = time.time()
    for dm, c_num in zip(domains, clients_per_domain):
        pool = domain_test[dm]
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
        logging.info(f"  域 {dm}：pool={n} → splits sizes {[len(sub) for sub in test_splits[-c_num:]]}")
    logging.info(f"[4/5] 完成测试集划分，总耗时 {time.time()-t4:.1f}s")

    # 5) 打印每个客户端的分布
    logging.info("[5/5] 统计/打印每个客户端的标签分布")
    train_dists = get_label_distribution(train_splits, num_classes=7)
    test_dists  = get_label_distribution(test_splits,  num_classes=7)
    for cid, (dm, td, vd) in enumerate(zip(client_domains, train_dists, test_dists), start=1):
        logging.info(f" Client{cid:02d} | Domain={dm:12s} | TrainDist={td} | TestDist={vd}")

    # 聚类（仅记录/保存，不影响划分）
    communities = adaptive_cluster_clients(train_dists)
    save_cluster_results(
        os.path.join(data_root, "peer_communities.json"),
        train_dists, test_dists, communities
    )
    logging.info(f"=== 全部完成，总耗时 {time.time()-t0:.1f}s ===")

    # Pack into FederatedScope
    client_data = {}
    for cid, (tr, te) in enumerate(zip(train_splits, test_splits), start=1):
        client_data[cid] = ClientData(client_cfg=config, train=tr, val=None, test=te)
    fsdata = StandaloneDataDict(client_data, config)
    return fsdata, config
def call_pacs_data(config, client_cfgs=None):
    if config.data.type.lower() in ("pacs",):
        return load_pacs(config, client_cfgs)

if __name__ == "__main__":
    from federatedscope.core.configs.config import global_cfg
    global_cfg.data.root = "~/FederatedScope/data/PACS"
    global_cfg.data.type = "pacs"
    data, cfg = load_pacs(global_cfg)
else:
    register_data("pacs111", call_pacs_data)
