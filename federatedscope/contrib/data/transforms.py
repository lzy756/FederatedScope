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
