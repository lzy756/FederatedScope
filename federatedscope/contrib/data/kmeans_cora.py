import copy
import numpy as np
import torch
from torch_geometric.datasets import Planetoid
from sklearn.cluster import KMeans
from federatedscope.core.data import DummyDataTranslator
from federatedscope.register import register_data

def cora_kmeans_partition(config=None, client_cfgs=None):
    # 1. 加载 Cora 并保留原始掩码
    path = config.data.root
    dataset = Planetoid(path, 'Cora', split='public')
    full_data = dataset[0]

    # 2. 基于节点特征做 KMeans 分配
    features = full_data.x.cpu().numpy()
    client_num = config.federate.client_num
    kmeans = KMeans(n_clusters=client_num, random_state=123)
    cluster_labels = kmeans.fit_predict(features)

    # 3. 按簇标签收集节点 ID
    client_node_ids = {i: [] for i in range(client_num)}
    for node_id, label in enumerate(cluster_labels):
        client_node_ids[label].append(node_id)

    # 4. 构造各客户端子图并本地 80/10/10 拆分
    client_graphs = {}
    for i in range(client_num):
        nodes = sorted(client_node_ids[i])
        if not nodes:
            continue
        subset = torch.tensor(nodes, dtype=torch.long)
        sub = full_data.subgraph(subset)

        n = sub.num_nodes
        perm = torch.randperm(n)
        n_train = max(int(0.8 * n), 1)
        n_val   = max(int(0.1 * n), 1)
        n_test  = n - n_train - n_val
        if n_test < 1:
            n_test = 1
            n_train = n - n_val - n_test

        t_idx = perm[:n_train]
        v_idx = perm[n_train:n_train + n_val]
        e_idx = perm[n_train + n_val:]

        sub.train_mask = torch.zeros(n, dtype=torch.bool)
        sub.val_mask   = torch.zeros(n, dtype=torch.bool)
        sub.test_mask  = torch.zeros(n, dtype=torch.bool)
        sub.train_mask[t_idx] = True
        sub.val_mask[v_idx]   = True
        sub.test_mask[e_idx]  = True

        client_graphs[i] = sub

    # 5. 打印调试信息：客户端标签分布
    print("=== KMeans 划分后客户端数据分布 ===")
    for i in range(client_num):
        print(f"Client {i+1}:")
        sub = client_graphs.get(i, None)
        if sub is None:
            print("  NO NODES")
            continue
        y = sub.y.cpu().numpy()
        def dist(mask):
            arr = y[mask.cpu().numpy()]
            u, c = np.unique(arr, return_counts=True)
            return dict(zip(u.tolist(), c.tolist()))
        print(f"  Train ({int(sub.train_mask.sum())}): {dist(sub.train_mask)}")
        print(f"  Val   ({int(sub.val_mask.sum())}): {dist(sub.val_mask)}")
        print(f"  Test  ({int(sub.test_mask.sum())}): {dist(sub.test_mask)}")
    print("===================================")

    # 6. 构造服务器端 full graph
    global_data = copy.deepcopy(full_data)

    # 7. 组装 data_local_dict
    data_local_dict = {}
    # 客户端 1..N
    for i, sub in client_graphs.items():
        data_local_dict[i+1] = {
            'data':  sub,
            'train': [sub],
            'val':   [sub],
            'test':  [sub],
        }
    # 服务器端 Client #0
    data_local_dict[0] = {
        'data':  global_data,
        'train': [global_data],
        'val':   [global_data],
        'test':  [global_data],
    }

    translator = DummyDataTranslator(config)
    return translator(data_local_dict), config


def call_cora_kmeans(config, client_cfgs=None):
    if config.data.type == "cora_kmeans":
        return cora_kmeans_partition(config, client_cfgs)

register_data("cora_kmeans", call_cora_kmeans)
