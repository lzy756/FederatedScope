# federatedscope/contrib/aggregator/fedsak_aggregator.py
import torch
from torch import nn
from federatedscope.core.aggregators import Aggregator


def _stack_param_matrix(client_states, key):
    """把同名权重展平成二维矩阵，维度 = (n_client, -1)"""
    mats = [state[key].flatten() for state in client_states]
    return torch.stack(mats, dim=0)  # [M, D]


def _unstack_to_clients(matrix, shapes):
    """反拆回各客户端张量形状"""
    mats = torch.split(matrix, 1, dim=0)  # tuple len=M
    mats = [m.squeeze(0) for m in mats]
    tensors = [m.reshape(shape) for m, shape in zip(mats, shapes)]
    return tensors


class FedSAKAggregator(Aggregator):
    r"""
    实现论文 FedSAK ⟨Liao et al., NeurIPS 2024⟩ 的核心：
    • 堆叠各客户端共享层 → 施加张量 trace‑norm 正则 → 次梯度更新
    • 返回 **个性化** slice 给对应客户端
    """
    def __init__(self, config, **kwargs):
        # super().__init__(config, **kwargs)
        self.lmbda = config.aggregator.get("lambda", 1e-3)
        self.lr = config.aggregator.get("lr_shared", 0.1)
        # 与 Trainer 保持同一顺序的参数名
        self.filter_keys = config.aggregator.get("filter_keys", [])

    # ------ 核心聚合函数 ------
    def aggregate(self, agg_info):
        """
        agg_info['client_feedback'] = [
            {'num_sample': n_i, 'model_para': {...}}, ...]
        """
        client_states = [
            fb['model_para'] for fb in agg_info['client_feedback']
        ]
        n_client = len(client_states)
        updated_dicts = [dict() for _ in range(n_client)]

        for key in self.filter_keys:
            # 1. 组装矩阵 W ∈ ℝ^{M×D}
            shapes = [state[key].shape for state in client_states]
            W = _stack_param_matrix(client_states, key)  # [M, D]

            # 2. Trace‑norm 次梯度  ‖W‖_*  =  ‖Σ‖_1
            #    W = U Σ V^T, 次梯度 = U V^T
            U, _, Vh = torch.linalg.svd(W, full_matrices=False)
            grad = torch.matmul(U, Vh)  # [M, D]

            # 3. Slice‑wise 更新
            W_new = W - self.lr * self.lmbda * grad

            # 4. 拆回客户端形状
            tensors = _unstack_to_clients(W_new, shapes)
            for i, t in enumerate(tensors):
                updated_dicts[i][key] = t

        # -------- 组织返回格式 --------
        # FederatedScope 允许返回 {"model_para_all": {cid: {...}}}
        cid_list = [fb['client_ID'] for fb in agg_info['client_feedback']]
        personalized = {
            cid: state
            for cid, state in zip(cid_list, updated_dicts)
        }
        return {"model_para_all": personalized}
