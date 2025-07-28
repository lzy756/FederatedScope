import logging
import copy
import torch
import numpy as np
from scipy.optimize import minimize

from federatedscope.core.aggregators import ClientsAvgAggregator
from federatedscope.core.auxiliaries.utils import param2tensor

logger = logging.getLogger(__name__)


class MIXAggregator(ClientsAvgAggregator):
    """MIX 聚合器实现

    继承自 ClientsAvgAggregator，实现 MIX 的分组聚合逻辑
    提供组内聚合和组间聚合两个核心方法：
    - aggregate_intra_group: 组内客户端模型聚合
    - aggregate_inter_group: 组间模型聚合，使用trace-norm正则化
    """

    def __init__(self, model=None, device="cpu", config=None):
        super().__init__(model, device, config)

        # 聚合参数配置
        self.lambda_ = config.aggregator.get("lambda_", 1e-3)  # trace-norm正则化强度
        self.intra = config.aggregator.get("lr_shared", 0)  # 是否启动组内共识优化

    def aggregate_intra_group(self, client_feedback):
        """组内聚合：聚合组内客户端的模型

        Args:
            client_feedback: 组内客户端反馈列表，每个元素包含:
                - client_id: 客户端ID
                - sample_size: 样本大小
                - model_para: 模型参数

        Returns:
            dict: 聚合后的模型参数
        """
        logger.info(
            f"Server: Performing intra-group aggregation with {len(client_feedback)} clients"
        )
        return self._intra_group_aggregate(client_feedback)

    def aggregate_inter_group(self, group_feedback):
        """组间聚合：使用trace-norm正则化方法聚合所有组的模型

        Args:
            group_feedback: 所有组的反馈列表，每个元素包含:
                - client_id: 组ID
                - sample_size: 组内总样本大小
                - model_para: 组内聚合后的模型参数

        Returns:
            dict: 包含个性化模型参数的字典，格式为 {"model_para_all": {group_id: model_params}}
        """
        logger.info(
            f"Server: Performing inter-group aggregation with {len(group_feedback)} groups"
        )
        return self._inter_group_aggregate(group_feedback)

    def _intra_group_aggregate(self, models):
        """组内聚合：使用共识最大化聚合组内客户端的模型（逐层处理）"""
        if len(models) == 1:
            return copy.deepcopy(models[0]["model_para"])

        aggregated_model = copy.deepcopy(models[0]["model_para"])
        layer_keys = list(aggregated_model.keys())
        # logger.info(f"Aggregating layers: {layer_keys}")

        # 逐层处理
        for layer_name in layer_keys:
            layer_vectors = []
            valid_norms = []

            # 根据配置选择聚合策略
            if self.intra == 0:
                # 使用简单平均聚合
                logger.debug(f"Using simple averaging for layer {layer_name}")
                layer_vectors = [
                    model["model_para"][layer_name].flatten()
                    for model in models
                    if layer_name in model["model_para"]
                ]
                if layer_vectors:
                    aggregated_model[layer_name] = torch.mean(
                        torch.stack(layer_vectors), dim=0
                    ).reshape(aggregated_model[layer_name].shape)
                else:
                    logger.info(f"No valid vectors found for layer {layer_name}")
                continue

            # 使用共识最大化聚合
            logger.debug(f"Using consensus maximization for layer {layer_name}")

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
                    logger.info(f"Layer {layer_name} missing in client model")
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
            logger.info(f"Optimization error: {str(e)}")

        # 回退方案：使用简单平均
        logger.info("Falling back to uniform weights")
        return [1.0 / N] * N

    def _inter_group_aggregate(self, intra_models):
        """组间聚合：使用trace-norm正则化方法聚合所有组的模型

        Args:
            intra_models: 所有组的模型列表

        Returns:
            dict: 包含个性化模型参数的字典
        """
        if self.lambda_ == 0:
            # 如果lambda为0，直接使用加权平均
            models = []
            for feedback in intra_models:
                sample_size, model_para = (
                    feedback["sample_size"],
                    feedback["model_para"],
                )
                models.append((sample_size, model_para))
            # 直接返回平均模型
            avg_model = self._para_weighted_avg(models)
            # 构建返回格式 - 为每个客户端创建独立的模型副本
            personalized = {
                feedback["client_id"]: copy.deepcopy(avg_model)
                for feedback in intra_models
            }
            return {"model_para_all": personalized}
        else:

            client_ids = []
            model_params = []

            for feedback in intra_models:
                client_ids.append(feedback["client_id"])
                model_para = feedback["model_para"]
                model_params.append(model_para)

            n_client = len(model_params)
            updated_dicts = [dict() for _ in range(n_client)]
            layers = list(model_params[0].keys())

            for layer in layers:
                logger.info(f"Processing layer: {layer}")
                if not all(layer in params for params in model_params):
                    logger.info(f"Layer {layer} missing in some models")
                    # 如果某个组没有此层，保留原参数（创建副本）
                    for idx in range(n_client):
                        if layer in model_params[idx]:
                            updated_dicts[idx][layer] = (
                                model_params[idx][layer].detach().clone()
                            )
                    continue
                shapes = [params[layer].shape for params in model_params]
                W = torch.stack([params[layer].flatten() for params in model_params], 0)
                # SVD分解
                U, S, Vh = torch.linalg.svd(W, full_matrices=False)

                # 正确的trace-norm次梯度计算
                # trace-norm的次梯度: 对于满秩情况，次梯度就是 U @ Vh
                # 但我们需要处理数值稳定性问题
                
                # 检查奇异值，避免数值不稳定
                rank = torch.sum(S > 1e-8).item()  # 有效秩
                if rank == 0:
                    # 如果矩阵几乎为零，不进行更新
                    logger.warning(f"Layer {layer} has near-zero singular values, skipping update")
                    for idx in range(n_client):
                        updated_dicts[idx][layer] = model_params[idx][layer].detach().clone()
                    continue
                
                # 使用有效的奇异向量计算次梯度
                U_eff = U[:, :rank]
                Vh_eff = Vh[:rank, :]
                grad = U_eff @ Vh_eff  # 这确实是 U × V^T 的结果

                # 更新参数
                W_new = W - self.lambda_ * grad

                # 还原各组参数形状
                for idx, slice_vec in enumerate(W_new):
                    updated_dicts[idx][layer] = slice_vec.reshape(shapes[idx])

            # 构建返回格式
            personalized = {gid: state for gid, state in zip(client_ids, updated_dicts)}

            return {"model_para_all": personalized}
