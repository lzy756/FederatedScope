# 文件：federatedscope/contrib/data/cora_pruned.py
# ----------------- 主要区别 -----------------
# ① 新增 data.args.top_k_cross （int）控制每个节点保留多少条跨客户端边；
#    默认 0 == 旧版“完全断开”。
# ② 不再一刀切删除跨边，而是：
#      • 计算所有跨边的“强度”——用 dst 节点的度数近似；
#      • 对每个 src 节点，按强度降序保留 top-k；
#      • 其余跨边剪掉。
# ③ 其余流程（节点切分 / 子图 relabel / mask 投影 / DummyDataTranslator）
#    全部沿用旧版，保证与 FS 其它组件兼容。
# -------------------------------------------

import copy
import numpy as np
import torch
from torch_geometric.datasets import Planetoid
from torch_geometric.data import Data
from federatedscope.core.data import DummyDataTranslator
from federatedscope.register import register_data


def cora_pruned_partition(config=None, client_cfgs=None):
    # ---------- 0. 读超参数 ----------
    top_k_cross = getattr(config.data.args, "top_k_cross", 0)  # 每节点保留的跨边数
    root = getattr(config.data, "root", "./data/Cora")

    # ---------- 1. 载入 Cora ----------
    dataset = Planetoid(root=root, name="Cora", split="public")
    full_data = dataset[0]
    num_nodes = full_data.num_nodes
    edge_index = full_data.edge_index
    x_all, y_all = full_data.x, full_data.y
    train_mask_all = full_data.train_mask
    val_mask_all, test_mask_all = full_data.val_mask, full_data.test_mask

    # ---------- 2. 随机节点划分 ----------
    client_num = config.federate.client_num
    idx_all = np.random.permutation(num_nodes)
    base, rem = divmod(num_nodes, client_num)
    client_nodes = {}
    ptr = 0
    for cid in range(client_num):
        cnt = base + (1 if cid < rem else 0)
        client_nodes[cid] = idx_all[ptr : ptr + cnt].tolist()
        ptr += cnt

    # ---------- 3. 预计算节点度（用于粗略衡量“边强度”） ----------
    deg = torch.bincount(edge_index[0], minlength=num_nodes)

    # ---------- 4. 为每个客户端生成子图 ----------
    data_local_dict = {}
    src_np, dst_np = edge_index[0].cpu().numpy(), edge_index[1].cpu().numpy()

    # 构造 node → client id 的反向映射，便于判断跨边
    node2cid = {}
    for cid, nodes in client_nodes.items():
        for n in nodes:
            node2cid[n] = cid

    for cid in range(client_num):
        nodes_i = np.array(client_nodes[cid], dtype=np.int64)
        node_set = set(nodes_i.tolist())

        # -- 4.1 先把“内部边”都留下 --
        keep_mask = []
        cross_bucket = {}  # src_node → [(edge_id, strength), ...]
        for eid, (u, v) in enumerate(zip(src_np, dst_np)):
            cu, cv = node2cid[u], node2cid[v]
            if cu == cv == cid:
                keep_mask.append(True)
            elif cu == cid or cv == cid:
                # 跨客户端边，放进 bucket 供稍后按强度筛选
                src_node = u if cu == cid else v
                strength = int(deg[u] + deg[v])  # 度和作近似权重
                cross_bucket.setdefault(src_node, []).append((eid, strength))
                keep_mask.append(False)  # 先都标 False，后面选中的再 True
            else:
                keep_mask.append(False)

        keep_mask = np.array(keep_mask, dtype=bool)

        # -- 4.2 逐 src 节点保留 top-k 跨边 --
        if top_k_cross > 0:
            for src, lst in cross_bucket.items():
                # lst: [(eid, strength), ...]
                lst_sorted = sorted(lst, key=lambda t: -t[1])
                for eid, _ in lst_sorted[:top_k_cross]:
                    keep_mask[eid] = True

        # -- 4.3 根据 keep_mask 取边 & relabel --
        eids = np.nonzero(keep_mask)[0]
        edge_i = edge_index[:, eids]
        # relabel 节点
        mapping = {int(old): new for new, old in enumerate(nodes_i)}
        edge_i_relabel = edge_i.clone()
        for col in range(edge_i_relabel.size(1)):
            edge_i_relabel[0, col] = mapping[int(edge_i_relabel[0, col])]
            edge_i_relabel[1, col] = mapping[int(edge_i_relabel[1, col])]

        # -- 4.4 构造 PyG Data --
        sub = Data(
            x=x_all[nodes_i],
            y=y_all[nodes_i],
            edge_index=edge_i_relabel,
            train_mask=train_mask_all[nodes_i],
            val_mask=val_mask_all[nodes_i],
            test_mask=test_mask_all[nodes_i],
        )

        data_local_dict[cid + 1] = {"data": sub, "train": [sub], "val": [sub], "test": [sub]}

    # ---------- 5. 服务器端完整图 ----------
    data_local_dict[0] = {"data": full_data, "val": [full_data], "test": [full_data]}

    translator = DummyDataTranslator(config)
    fs_data, cfg = translator(data_local_dict), config
    return fs_data, cfg


def call_cora_pruned(config, client_cfgs=None):
    if config.data.type == "cora_pruned":
        return cora_pruned_partition(config, client_cfgs)


register_data("cora_pruned", call_cora_pruned)
