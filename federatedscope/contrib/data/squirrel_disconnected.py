# 文件：federatedscope/contrib/data/squirrel_disconnected.py

import copy
import numpy as np
import torch
from torch_geometric.datasets import WikipediaNetwork
from torch_geometric.data import Data
from federatedscope.core.data import DummyDataTranslator
from federatedscope.register import register_data


def squirrel_disconnected_partition(config=None, client_cfgs=None):
    """
    “完全断开”版的 Squirrel 划分：
      1. 用 PyG 的 WikipediaNetwork 去加载本地的 squirrel/raw/ 中的原始文件。
      2. 从 10 种固定 split 中，选出第 0 列作为 train/val/test 掩码（1D 布尔）。
      3. 对所有节点打乱并平均划分到各 client，然后对每个 client 只保留内部边，构造子图。
      4. 保证每个子图的 train_mask/val_mask/test_mask 均为 1D 布尔向量。
      5. 服务器端保留完整全图，用于全局评估。
    """

    # ——— 1. 让 PyG 加载本地已下载的 Squirrel raw 文件 ———
    #    假设：config.data.root = '/home/liziyu/FederatedScope/data/'，
    #    并且 raw 文件在 '/home/liziyu/FederatedScope/data/squirrel/raw/'。
    dataset = WikipediaNetwork(root=config.data.root, name='squirrel')
    full_data = dataset[0]   # PyG Data：包含 x, edge_index, y, train_mask(2D), val_mask(2D), test_mask(2D)

    # ——— 2. 只选 10 个 split 里的第 0 列，将 2D mask → 1D mask ———
    split_index = 1  # 也可以换成 1~9 中的任意一个，代表用不同的划分方案
    full_data.train_mask = full_data.train_mask[:, split_index]
    full_data.val_mask   = full_data.val_mask[:,   split_index]
    full_data.test_mask  = full_data.test_mask[:,  split_index]

    print("full_data.train_mask.shape:", full_data.train_mask.shape)
    print("full_data.val_mask.shape:  ", full_data.val_mask.shape)
    print("full_data.test_mask.shape: ", full_data.test_mask.shape)

    # 现在 train_mask, val_mask, test_mask 形状都变成了 [num_nodes]

    # 基本图信息
    num_nodes = full_data.num_nodes
    edge_index = full_data.edge_index   # [2, num_edges]
    x_all = full_data.x                 # [num_nodes, num_features]
    y_all = full_data.y                 # [num_nodes]

    # ——— 3. 随机打乱所有节点并平均分给 client_num 个客户端 ———
    client_num = config.federate.client_num
    all_indices = np.arange(num_nodes)
    np.random.shuffle(all_indices)

    base_count = num_nodes // client_num
    remainder  = num_nodes - base_count * client_num

    client_node_indices = {}
    ptr = 0
    for cid in range(client_num):
        cnt = base_count + (1 if cid < remainder else 0)
        client_node_indices[cid] = all_indices[ptr:ptr+cnt].tolist()
        ptr += cnt

    # ——— 4. 构造每个客户端的子图，只保留“内部边” & “投影 Mask” ———
    data_local_dict = {}
    for cid in range(client_num):
        nodes_i = np.array(client_node_indices[cid], dtype=np.int64)
        if nodes_i.size == 0:
            # 万一某客户端分配到 0 个节点，则构造空图
            sub_data = Data()
            sub_data.x = torch.empty((0, x_all.size(1)))
            sub_data.y = torch.empty((0,), dtype=y_all.dtype)
            sub_data.edge_index = torch.empty((2, 0), dtype=torch.long)
            sub_data.train_mask = torch.empty((0,), dtype=torch.bool)
            sub_data.val_mask   = torch.empty((0,), dtype=torch.bool)
            sub_data.test_mask  = torch.empty((0,), dtype=torch.bool)
            data_local_dict[cid+1] = {
                'data': sub_data,
                'train': [sub_data],
                'val':   [sub_data],
                'test':  [sub_data],
            }
            continue

        # 在 full_data.edge_index 上筛选“两端都在 nodes_i 里”的边
        src = edge_index[0].cpu().numpy()
        dst = edge_index[1].cpu().numpy()
        node_set = set(nodes_i.tolist())
        mask_bool = [(int(u) in node_set and int(v) in node_set)
                     for u, v in zip(src, dst)]
        mask_bool = np.array(mask_bool, dtype=bool)
        selected_edges = np.nonzero(mask_bool)[0]
        edge_index_i = edge_index[:, selected_edges]  # [2, Ei]

        # 节点重编号：将 global idx → [0..|nodes_i|-1]
        mapping = {int(nodes_i[j]): j for j in range(nodes_i.size)}
        edge_index_i_relabel = edge_index_i.clone()
        for eid in range(edge_index_i_relabel.size(1)):
            u_glob = int(edge_index_i_relabel[0, eid].item())
            v_glob = int(edge_index_i_relabel[1, eid].item())
            edge_index_i_relabel[0, eid] = mapping[u_glob]
            edge_index_i_relabel[1, eid] = mapping[v_glob]

        # 新建子图
        sub = Data()
        sub.x = x_all[nodes_i]        # [ni, num_features]
        sub.y = y_all[nodes_i]        # [ni]
        sub.edge_index = edge_index_i_relabel  # [2, Ei]

        # “投影”全局 1D mask → 局部 1D mask
        train_loc = full_data.train_mask.cpu().numpy()[nodes_i]
        val_loc   = full_data.val_mask.cpu().numpy()[nodes_i]
        test_loc  = full_data.test_mask.cpu().numpy()[nodes_i]

        sub.train_mask = torch.tensor(train_loc, dtype=torch.bool)  # [ni]
        sub.val_mask   = torch.tensor(val_loc,   dtype=torch.bool)  # [ni]
        sub.test_mask  = torch.tensor(test_loc,  dtype=torch.bool)  # [ni]

        # 保存到 data_local_dict，客户端编号从 1 开始
        data_local_dict[cid+1] = {
            'data': sub,
            'train': [sub],
            'val':   [sub],
            'test':  [sub],
        }

    # ——— 5. 服务器端保留原始全图，仅做全局评估用 ———
    data_local_dict[0] = {
        'data': full_data,
        'val':  [full_data],
        'test': [full_data],
    }

    # ——— 调试信息：查看每个客户端子图的 train/val/test 大小 ———
    print("=== 各客户端子图节点 & Mask 统计（断开连接） ===")
    for cid, dct in data_local_dict.items():
        if cid == 0:
            continue
        sub = dct['data']
        print(f" Client {cid}: "
              f"num_nodes={sub.num_nodes}, "
              f"train={int(sub.train_mask.sum())}, "
              f"val={int(sub.val_mask.sum())}, "
              f"test={int(sub.test_mask.sum())}")
    print("===========================================")

    # ——— 6. 转换为 FederatedScope 格式并返回 ———
    translator = DummyDataTranslator(config)
    fs_data, cfg = translator(data_local_dict), config
    return fs_data, cfg


def call_squirrel_disconnected(config, client_cfgs=None):
    if config.data.type == "squirrel_disconnected":
        return squirrel_disconnected_partition(config, client_cfgs)


register_data("squirrel_disconnected", call_squirrel_disconnected)
