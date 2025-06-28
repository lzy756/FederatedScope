"""
文件：federatedscope/contrib/data/citeseer_disconnected.py

描述：
    “完全断开”版的 CiteSeer 划分。先加载本地 CiteSeer 数据集，
    然后将所有节点随机分配到客户端，并构造每个客户端仅保留内部边的子图。

使用方法：
    1. 保存本文件到 `federatedscope/contrib/data/citeseer_disconnected.py`。
    2. 在你的 YAML 配置中指定：
         data:
           root: data/           # 上级目录，包含子目录 CiteSeer
           type: citeseer_disconnected
    3. 确保 federate.client_num 与客户端数量一致。
    4. 其他训练配置（model、trainer、eval）保持不变，运行即可。
"""

import os
import random
import numpy as np
import torch
from torch_geometric.datasets import Planetoid
from torch_geometric.data import Data
from torch_geometric.utils import k_hop_subgraph
from federatedscope.core.data import DummyDataTranslator
from federatedscope.register import register_data

def citeseer_disconnected_partition(config=None, client_cfgs=None):
    # 1. 加载 CiteSeer 全图
    root = getattr(config.data, 'root', './data/CiteSeer')
    dataset = Planetoid(root=root, name='CiteSeer', split='public')
    full_data = dataset[0]
    print(f"Total number of nodes in CiteSeer: {full_data.num_nodes}")
    # 参数
    num_clients = config.federate.client_num
    num_ego = getattr(config.data, 'num_ego', 300)
    k_hop = getattr(config.data, 'k_hop', 2)

    # 2. 随机采样 Ego 节点
    train_idx = full_data.train_mask.nonzero(as_tuple=False).view(-1).tolist()
    random.seed(42)
    ego_nodes = random.sample(train_idx, min(num_ego, len(train_idx)))

    # 3. 构建 Ego-Subgraphs
    subgraphs = []  # list of Data objects
    for ego in ego_nodes:
        subset, edge_index, mapping, hard_mask = k_hop_subgraph(
            node_idx=ego,
            num_hops=k_hop,
            edge_index=full_data.edge_index,
            relabel_nodes=True
        )
        sub = Data(
            x=full_data.x[subset],
            y=full_data.y[subset],
            edge_index=edge_index
        )
        # masks: inherit only nodes in subset
        sub.train_mask = full_data.train_mask[subset]
        sub.val_mask = full_data.val_mask[subset]
        sub.test_mask = full_data.test_mask[subset]
        subgraphs.append(sub)

    # 4. 将 Subgraphs 分配给客户端
    data_local_dict = {}
    random.shuffle(subgraphs)
    per_client = int(np.ceil(len(subgraphs) / num_clients))
    for cid in range(1, num_clients + 1):
        assigned = subgraphs[(cid - 1) * per_client: cid * per_client]
        if not assigned:
            assigned = [subgraphs[(cid - 1) % len(subgraphs)]]
        # 使用首个子图作为 data 用于 shape 推断
        data_local_dict[cid] = {'data': assigned[0], 'train': assigned, 'val': assigned, 'test': assigned}

    # 5. 打印各客户端数据分布统计
    print("=== 客户端数据分布统计 (CiteSeer Ego-Shard) ===")
    for cid, dct in data_local_dict.items():
        if cid == 0: continue
        train_count = sum([s.train_mask.sum().item() for s in dct['train']])
        val_count = sum([s.val_mask.sum().item() for s in dct['val']])
        test_count = sum([s.test_mask.sum().item() for s in dct['test']])
        print(
            f"Client {cid}: subgraphs={len(dct['train'])}, train_nodes={train_count}, val_nodes={val_count}, test_nodes={test_count}")
    print("==============================================")

    # 5. 服务器保留全局全图用于评估
    data_local_dict[0] = {'data': full_data, 'val': [full_data], 'test': [full_data]}

    # 6. 转换并返回
    translator = DummyDataTranslator(config)
    fs_data, cfg = translator(data_local_dict), config
    return fs_data, cfg


def call_citeseer_disconnected(config, client_cfgs=None):
    if config.data.type == "citeseer_disconnected":
        return citeseer_disconnected_partition(config, client_cfgs)

register_data("citeseer_disconnected", call_citeseer_disconnected)
