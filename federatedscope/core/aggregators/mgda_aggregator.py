# Multi-Task Learning：MGDA
import logging
import torch
from torch import nn
import copy
from scipy.optimize import minimize
import numpy as np
from federatedscope.core.aggregators import ClientsAvgAggregator
from federatedscope.core.auxiliaries.utils import param2tensor

logger = logging.getLogger(__name__)


class MGDAAggregator(ClientsAvgAggregator):
    """
    Implementation of Multi-Task Gradient Descent Aggregator (MGDA)
    """

    def __init__(self, model=None, device="cpu", config=None):
        super(MGDAAggregator, self).__init__(model=model, device=device, config=config)

    def aggregate(self, agg_info):
        """
        To preform aggregation

        Arguments:
            agg_info (dict): the feedbacks from clients

        Returns:
            dict: the aggregated results
        """
        models = agg_info["client_feedback"]

        return {"model": self._intra_group_aggregate(models)}

    def _intra_group_aggregate(self, models):
        """组内聚合：使用共识最大化聚合组内客户端的模型（逐层处理）"""
        if len(models) == 1:
            return models[0]["model_para"]

        logger.info(
            f"Performing LAYER-WISE consensus maximization with {len(models)} clients"
        )

        aggregated_model = copy.deepcopy(models[0]["model_para"])
        layer_keys = list(aggregated_model.keys())
        logger.info(f"Aggregating layers: {layer_keys}")

        # 逐层处理
        for layer_name in layer_keys:
            layer_vectors = []
            valid_norms = []

            # 收集当前层所有客户端的参数
            for model in models:
                if layer_name in model["model_para"]:
                    layer_param = model["model_para"][layer_name]
                    layer_vector = layer_param.flatten()

                    # 计算并保存该层的L2范数
                    layer_norm = torch.norm(layer_vector, p=2)
                    if layer_norm > 1e-8:  # 避免除零
                        valid_norms.append(layer_norm)
                        layer_vectors.append(layer_vector / layer_norm)
                    else:
                        # 范数接近零，保持原向量（避免信息丢失）
                        layer_vectors.append(layer_vector.clone())
                else:
                    logger.warning(f"Layer {layer_name} missing in client model")
                    layer_shape = aggregated_model[layer_name].shape
                    layer_vectors.append(
                        torch.zeros(
                            torch.prod(torch.tensor(layer_shape)).to(
                                layer_vectors[0].device
                            )
                        )
                    )

            # 当前层执行共识最大化
            if layer_vectors:
                # 计算平均范数（只使用有效范数）
                avg_norm = sum(valid_norms) / len(valid_norms) if valid_norms else 0

                # 优化权重（使用支持等式约束的SLSQP方法）
                weights = self._solve_consensus_weights(layer_vectors)

                # 应用权重聚合
                aggregated_vector = torch.zeros_like(layer_vectors[0])
                for i, vector in enumerate(layer_vectors):
                    aggregated_vector += weights[i] * vector

                # 恢复原始尺度
                if avg_norm > 1e-8:  # 仅在有效范数时缩放
                    aggregated_vector *= avg_norm

                aggregated_model[layer_name] = aggregated_vector.reshape(
                    aggregated_model[layer_name].shape
                )

                logger.info(
                    f"Layer {layer_name} aggregated | Avg norm: {avg_norm:.4f} | Weights: {weights}"
                )

        return aggregated_model

    def _solve_consensus_weights(self, normalized_vectors):
        """支持等式约束的共识权重优化"""
        N = len(normalized_vectors)
        if N == 1:
            return [1.0]

        # 转换为numpy数组
        updates_np = [vec.detach().cpu().numpy() for vec in normalized_vectors]

        def objective(u):
            """目标函数：最小化加权和的L2范数平方"""
            weighted_sum = np.zeros_like(updates_np[0])
            for i, weight in enumerate(u):
                weighted_sum += weight * updates_np[i]
            return np.linalg.norm(weighted_sum) ** 2

        # 初始权重（均匀分布）
        u0 = np.ones(N) / N

        # 约束配置：权重和为1（等式约束）
        constraints = [{"type": "eq", "fun": lambda u: np.sum(u) - 1.0}]
        bounds = [(0.0, 1.0) for _ in range(N)]  # 权重范围0-1

        try:
            # 使用SLSQP方法（支持等式约束）
            result = minimize(
                objective,
                u0,
                method="SLSQP",
                bounds=bounds,
                constraints=constraints,
                options={"maxiter": 500, "ftol": 1e-8},
            )

            if result.success:
                weights = result.x / np.sum(result.x)  # 确保归一化
                weights = np.maximum(weights, 0)  # 确保非负
                return weights.tolist()
        except Exception as e:
            logger.warning(f"Optimization error: {str(e)}")

        # 回退方案：使用简单平均
        logger.warning("Falling back to uniform weights")
        return [1.0 / N] * N
