import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from federatedscope.register import register_model

class DebugGCN(torch.nn.Module):
    def __init__(self, in_feats, hidden, num_classes, dropout):
        super().__init__()
        self.conv1 = GCNConv(in_feats, hidden)
        self.conv2 = GCNConv(hidden, num_classes)
        self.dropout = dropout

    def forward(self, x, edge_index):
        # 打印输入
        print(f"[DEBUG][Model] x.shape={x.shape}, edge_index.shape={edge_index.shape}")
        # 第一层
        x1 = self.conv1(x, edge_index)
        print(f"[DEBUG][Model] after conv1: shape={x1.shape}, mean={x1.mean().item():.4f}, std={x1.std().item():.4f}")
        x1 = F.relu(x1)
        x1 = F.dropout(x1, p=self.dropout, training=self.training)
        # 第二层
        x2 = self.conv2(x1, edge_index)
        print(f"[DEBUG][Model] after conv2: shape={x2.shape}, mean={x2.mean().item():.4f}, std={x2.std().item():.4f}")
        out = F.log_softmax(x2, dim=-1)
        return out

def call_debug_gcn(cfg, input_shape):
    # input_shape: ((num_nodes, feature_dim), num_classes, ...)
    in_feats = input_shape[0][1]
    return DebugGCN(
        in_feats=in_feats,
        hidden=cfg.model.hidden,
        num_classes=cfg.model.num_classes,
        dropout=cfg.model.dropout
    )

register_model('debug_gcn', call_debug_gcn)
