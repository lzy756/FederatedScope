""
"""
文件：federatedscope/contrib/data/ogbn_arxiv_disconnected.py

描述：
    基于 Ego-Graph 分片的 OGBN-Arxiv 异构划分。
    先从本地 CSV 文件加载完整图数据（节点特征、标签、时间切分）；
    再从训练节点中随机采样若干中心节点（egos），构建 k-hop 子图；
    将这些子图等分分配给不同客户端，并打印每个客户端的子图及节点统计。

使用方法：
    1. 保存本文件到 `federatedscope/contrib/data/ogbn_arxiv_disconnected.py`。
    2. 确保原始 CSV 文件位于：
         <config.data.root>/ogbn_arxiv/data/ogbn_arxiv/ogbn_arxiv/raw/
       包含：edge.csv.gz, node-feat.csv.gz, node-label.csv.gz, node_year.csv.gz
    3. 在你的 YAML 配置中指定：
         data:
           root: data/             # 上级目录，含子目录 ogbn_arxiv/data/ogbn_arxiv/ogbn_arxiv/raw
           type: ogbn_arxiv_disconnected
           num_ego: 500            # 采样的 Ego 数量
           k_hop: 2                # hop 数
    4. 设置 `federate.client_num` 为客户端数量。
    5. 其他训练配置保持不变，运行即可。
"""

import os
import random
import gzip
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.utils import k_hop_subgraph
from federatedscope.core.data import DummyDataTranslator
from federatedscope.register import register_data

def ogbn_arxiv_disconnected_partition(config=None, client_cfgs=None):
    # 1. 加载本地 CSV 数据
    base_root = getattr(config.data, 'root', './data')
    raw_dir = os.path.join(base_root, 'ogbn_arxiv', 'data', 'ogbn_arxiv', 'ogbn_arxiv', 'raw')
    # 节点特征
    feat_path = os.path.join(raw_dir, 'node-feat.csv.gz')
    with gzip.open(feat_path, 'rt') as f:
        feats = np.loadtxt(f, delimiter=',')
    x = torch.tensor(feats, dtype=torch.float)
    # 节点标签
    label_path = os.path.join(raw_dir, 'node-label.csv.gz')
    with gzip.open(label_path, 'rt') as f:
        labels = np.loadtxt(f, dtype=int, delimiter=',')
    y = torch.tensor(labels, dtype=torch.long).view(-1)
    # 节点年份（用于划分 train/val/test）
    year_path = os.path.join(raw_dir, 'node_year.csv.gz')
    with gzip.open(year_path, 'rt') as f:
        years = np.loadtxt(f, dtype=int, delimiter=',')
    # 训练/验证/测试 mask
    num_nodes = x.shape[0]
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask   = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask  = torch.zeros(num_nodes, dtype=torch.bool)
    train_mask[years <= 2017] = True
    val_mask[years == 2018]   = True
    test_mask[years >= 2019]  = True
    # 边列表
    edge_path = os.path.join(raw_dir, 'edge.csv.gz')
    with gzip.open(edge_path, 'rt') as f:
        edges = np.loadtxt(f, dtype=int, delimiter=',')
    edge_index = torch.tensor(edges.T, dtype=torch.long)

    # 构造原始全图
    full_data = Data(x=x, y=y, edge_index=edge_index)
    full_data.train_mask = train_mask
    full_data.val_mask   = val_mask
    full_data.test_mask  = test_mask
    print(f"Loaded OGBN-Arxiv with {num_nodes} nodes, {edge_index.size(1)} edges.")

    # 参数
    num_clients = config.federate.client_num
    num_ego = getattr(config.data, 'num_ego', 500)
    k_hop = getattr(config.data, 'k_hop', 2)

    # 2. 从训练节点随机采样 Ego
    train_idx = full_data.train_mask.nonzero(as_tuple=False).view(-1).tolist()
    random.seed(42)
    ego_nodes = random.sample(train_idx, min(num_ego, len(train_idx)))

    # 3. 构建 k-hop 子图
    subgraphs = []
    for ego in ego_nodes:
        subset, ei, mapping, hard_mask = k_hop_subgraph(
            node_idx=ego,
            num_hops=k_hop,
            edge_index=full_data.edge_index,
            relabel_nodes=True
        )
        sub = Data(
            x=full_data.x[subset],
            y=full_data.y[subset],
            edge_index=ei
        )
        sub.train_mask = full_data.train_mask[subset]
        sub.val_mask   = full_data.val_mask[subset]
        sub.test_mask  = full_data.test_mask[subset]
        subgraphs.append(sub)

    # 4. 分配子图到客户端
    data_local_dict = {}
    random.shuffle(subgraphs)
    per_client = int(np.ceil(len(subgraphs) / num_clients))
    for cid in range(1, num_clients+1):
        assigned = subgraphs[(cid-1)*per_client : cid*per_client]
        if not assigned:
            assigned = [subgraphs[(cid-1) % len(subgraphs)]]
        data_local_dict[cid] = {
            'data': assigned[0],
            'train': assigned,
            'val':   assigned,
            'test':  assigned
        }

    # 5. 打印数据分布
    print("=== 客户端数据分布 (OGBN-Arxiv Ego-Shard) ===")
    for cid, dct in data_local_dict.items():
        train_n = sum([s.train_mask.sum().item() for s in dct['train']])
        val_n   = sum([s.val_mask.sum().item()   for s in dct['val']])
        test_n  = sum([s.test_mask.sum().item()  for s in dct['test']])
        print(f"Client {cid}: subgraphs={len(dct['train'])}, train_nodes={train_n}, val_nodes={val_n}, test_nodes={test_n}")
    print("============================================")

    # 6. 服务器保留全图
    data_local_dict[0] = {'data': full_data, 'val': [full_data], 'test': [full_data]}

    # 7. 转换为 FederatedScope 格式
    translator = DummyDataTranslator(config)
    fs_data, cfg = translator(data_local_dict), config
    return fs_data, cfg


def call_ogbn_arxiv_disconnected(config, client_cfgs=None):
    if config.data.type == "ogbn_arxiv_disconnected":
        return ogbn_arxiv_disconnected_partition(config, client_cfgs)

register_data("ogbn_arxiv_disconnected", call_ogbn_arxiv_disconnected)
""
