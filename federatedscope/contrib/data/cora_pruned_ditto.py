# 文件：federatedscope/contrib/data/cora_pruned_globalval.py
# -----------------------------------------------------------
# ✅  与 cora_pruned.py 共用“剪边 + Dirichlet 划分”逻辑
# ✅  但每个客户端的 `val` / `test` 直接引用 full_data
# -----------------------------------------------------------

import numpy as np
import torch
from torch_geometric.datasets import Planetoid
from torch_geometric.data import Data
from federatedscope.core.data import DummyDataTranslator
from federatedscope.register import register_data


def cora_pruned_gval_partition(config=None, client_cfgs=None):
    # ---------- 参数 ----------
    top_k_cross = getattr(config.data.args, "top_k_cross", 0)
    root = getattr(config.data, "root", "./data/Cora")

    # ---------- 读 Cora ----------
    dataset = Planetoid(root=root, name="Cora", split="public")
    full_data = dataset[0]
    num_nodes = full_data.num_nodes
    edge_index = full_data.edge_index
    x_all, y_all = full_data.x, full_data.y
    train_mask_all = full_data.train_mask
    val_mask_all, test_mask_all = full_data.val_mask, full_data.test_mask

    # ---------- Dirichlet 节点划分 ----------
    client_num = config.federate.client_num
    idx_all = np.random.permutation(num_nodes)
    base, rem = divmod(num_nodes, client_num)
    client_nodes = {}
    ptr = 0
    for cid in range(client_num):
        cnt = base + (1 if cid < rem else 0)
        client_nodes[cid] = idx_all[ptr : ptr + cnt].tolist()
        ptr += cnt

    # ---------- 预计算度数 ----------
    deg = torch.bincount(edge_index[0], minlength=num_nodes)

    # ---------- 构造子图 ----------
    data_local_dict = {}
    src_np, dst_np = edge_index[0].cpu().numpy(), edge_index[1].cpu().numpy()
    node2cid = {n: cid for cid, lst in client_nodes.items() for n in lst}

    for cid in range(client_num):
        nodes_i = np.array(client_nodes[cid], dtype=np.int64)
        node_set = set(nodes_i.tolist())

        keep_mask = []
        cross_bucket = {}
        for eid, (u, v) in enumerate(zip(src_np, dst_np)):
            cu, cv = node2cid[u], node2cid[v]
            if cu == cv == cid:
                keep_mask.append(True)
            elif cu == cid or cv == cid:
                src_node = u if cu == cid else v
                strength = int(deg[u] + deg[v])
                cross_bucket.setdefault(src_node, []).append((eid, strength))
                keep_mask.append(False)
            else:
                keep_mask.append(False)
        keep_mask = np.array(keep_mask, dtype=bool)

        if top_k_cross > 0:
            for src, lst in cross_bucket.items():
                for eid, _ in sorted(lst, key=lambda t: -t[1])[:top_k_cross]:
                    keep_mask[eid] = True

        eids = np.nonzero(keep_mask)[0]
        edge_i = edge_index[:, eids]

        # relabel
        mapping = {int(old): new for new, old in enumerate(nodes_i)}
        edge_i_rel = edge_i.clone()
        for col in range(edge_i_rel.size(1)):
            edge_i_rel[0, col] = mapping[int(edge_i_rel[0, col])]
            edge_i_rel[1, col] = mapping[int(edge_i_rel[1, col])]

        sub = Data(
            x=x_all[nodes_i],
            y=y_all[nodes_i],
            edge_index=edge_i_rel,
            train_mask=train_mask_all[nodes_i],
            # 子图内部 val/test 可留空或裁剪；这里保留但不会用
            val_mask=val_mask_all[nodes_i],
            test_mask=test_mask_all[nodes_i],
        )

        # ---------------- 关键：val/test = full_data ----------------
        data_local_dict[cid + 1] = {
            "data":  sub,                # 用来训练
            "train": [sub],
            "val":   [full_data],        # 用全局验证
            "test":  [full_data],        # 用全局测试
        }

    # ---------- 服务器端是否保留全集？ ----------
    # 若不想服务器评估，可留空：
    data_local_dict[0] = {"data": full_data}
    # 如果你仍想服务器做 make_global_eval=True 的对照，
    # 改成：
    # data_local_dict[0] = {"data": full_data,
    #                       "val":  [full_data],
    #                       "test": [full_data]}

    translator = DummyDataTranslator(config)
    fs_data, cfg = translator(data_local_dict), config
    return fs_data, cfg


def call_cora_pruned_globalval(config, client_cfgs=None):
    if config.data.type == "cora_pruned_globalval":
        return cora_pruned_gval_partition(config, client_cfgs)


register_data("cora_pruned_globalval", call_cora_pruned_globalval)
