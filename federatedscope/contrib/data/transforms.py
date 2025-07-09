# federatedscope/contrib/data/transforms.py

import torch
from torch_geometric.transforms import BaseTransform

class PadFeatures(BaseTransform):
    """
    将每个图的 data.x pad 或截断到固定维度 target_dim。
    """
    def __init__(self, target_dim: int):
        self.target_dim = target_dim

    def __call__(self, data):
        x = data.x
        C = x.size(1)
        if C < self.target_dim:
            pad = self.target_dim - C
            padding = x.new_zeros(x.size(0), pad)
            data.x = torch.cat([x, padding], dim=1)
        elif C > self.target_dim:
            data.x = x[:, :self.target_dim]
        return data

class AddEdgeAttr(BaseTransform):
    """
    给每个 Data 对象添加 edge_attr，如果它不存在的话。
    生成 shape=(num_edges, 0) 的空 tensor 占位。
    """
    def __call__(self, data):
        # PyG Data 可能用 _store 存储属性
        if not hasattr(data, 'edge_attr'):
            num_e = data.edge_index.size(1)
            data.edge_attr = torch.zeros((num_e, 0), dtype=torch.float)
        return data