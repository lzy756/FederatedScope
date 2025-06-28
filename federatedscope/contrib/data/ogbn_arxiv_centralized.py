"""
文件：federatedscope/contrib/data/ogbn_arxiv_centralized_debug.py

描述：
    将本地 OGBN-Arxiv 全图加载到单一客户端（模拟集中式训练），
    并打印详细的调试信息以排查数据和模型输入输出问题。

使用方法：
    1. 保存本文件到 `federatedscope/contrib/data/ogbn_arxiv_centralized_debug.py`。
    2. 在实验 YAML 中指定：
         data:
           root: data/
           type: ogbn_arxiv_centralized_debug
         federate:
           client_num: 1
    3. 其它配置同集中式脚本，运行时会在控制台打印调试信息。
"""

import os
import pandas as pd
import torch
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected, add_self_loops
from federatedscope.core.data import DummyDataTranslator
from federatedscope.register import register_data

def read_ogbn_arxiv_debug(root):
    """
    从本地 CSV 读取 OGBN-Arxiv 数据，构建 PyG Data，
    并打印基本的调试信息。
    """
    raw_dir = os.path.join(
        root,
        'ogbn_arxiv', 'data', 'ogbn_arxiv', 'ogbn_arxiv', 'raw'
    )

    # 1. 读取边列表
    edge_path = os.path.join(raw_dir, 'edge.csv.gz')
    edge_df = pd.read_csv(edge_path, compression='gzip', header=None, names=['src', 'dst'])
    src = torch.from_numpy(edge_df['src'].values).long()
    dst = torch.from_numpy(edge_df['dst'].values).long()
    edge_index = torch.stack([src, dst], dim=0)
    print(f"[DEBUG] 原始边数: {edge_index.size(1)}")

    # 对称化 & 自环
    ei_und = to_undirected(edge_index)
    print(f"[DEBUG] 对称化后边数: {ei_und.size(1)}")
    ei_loop, _ = add_self_loops(ei_und, num_nodes=ei_und.max().item() + 1)
    print(f"[DEBUG] 加自环后边数: {ei_loop.size(1)}")
    edge_index = ei_loop

    # 2. 读取节点特征
    feat_path = os.path.join(raw_dir, 'node-feat.csv.gz')
    feat_df = pd.read_csv(feat_path, compression='gzip', header=None)
    x = torch.from_numpy(feat_df.values).float()
    print(f"[DEBUG] 特征矩阵尺寸: {x.size()}")

    # 3. 读取标签
    label_path = os.path.join(raw_dir, 'node-label.csv.gz')
    label_df = pd.read_csv(label_path, compression='gzip', header=None)
    y = torch.from_numpy(label_df.values).view(-1).long()
    print(f"[DEBUG] 标签向量尺寸: {y.size()}")
    print(f"[DEBUG] 标签最小/最大: {y.min().item()}/{y.max().item()}")

    # 统计每类样本数（假设 num_classes=40）
    num_classes = y.max().item() + 1
    hist = torch.bincount(y, minlength=num_classes)
    print(f"[DEBUG] 各类样本数: {hist.tolist()}")

    # 4. 读取年份并构造 Mask
    year_path = os.path.join(raw_dir, 'node_year.csv.gz')
    year_df = pd.read_csv(year_path, compression='gzip', header=None)
    years = year_df[0].values
    train_mask = torch.from_numpy((years <= 2017)).bool()
    val_mask   = torch.from_numpy((years == 2018)).bool()
    test_mask  = torch.from_numpy((years >= 2019)).bool()
    print(f"[DEBUG] Mask 划分 (train/val/test): "
          f"{train_mask.sum().item()}/"
          f"{val_mask.sum().item()}/"
          f"{test_mask.sum().item()}")

    data = Data(
        x=x,
        edge_index=edge_index,
        y=y,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask
    )
    return data

def ogbn_arxiv_centralized_debug_partition(config=None, client_cfgs=None):
    """
    加载全图到单客户端，并保留全图用于服务器评估。
    同时打印数据维度等调试信息。
    """
    root = getattr(config.data, 'root', './data')
    full_data = read_ogbn_arxiv_debug(root)

    # 打印全图基本信息
    print(f"[DEBUG] 全图节点数: {full_data.num_nodes}")
    print(f"[DEBUG] 全图边数  : {full_data.edge_index.size(1)}")

    # 构造单客户端数据
    data_local_dict = {
        1: {
            'data':  full_data,
            'train': [full_data],
            'val':   [full_data],
            'test':  [full_data],
        },
        0: {
            'data':  full_data,
            'val':   [full_data],
            'test':  [full_data],
        }
    }

    # 转换并返回
    translator = DummyDataTranslator(config)
    fs_data, cfg = translator(data_local_dict), config
    return fs_data, cfg

def call_ogbn_arxiv_centralized(config, client_cfgs=None):
    if config.data.type == "ogbn_arxiv_centralized":
        return ogbn_arxiv_centralized_debug_partition(config, client_cfgs)

register_data("ogbn_arxiv_centralized", call_ogbn_arxiv_centralized)
