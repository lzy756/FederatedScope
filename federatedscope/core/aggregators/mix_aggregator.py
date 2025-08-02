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

        sample_sizes = [model["sample_size"] for model in models]
        total_samples = sum(sample_sizes)
        weights_on_samples = [size / total_samples for size in sample_sizes]

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
                    # 根据weights_on_samples计算加权平均
                    weighted_sum = sum(
                        weight * vec
                        for weight, vec in zip(weights_on_samples, layer_vectors)
                    )
                    aggregated_model[layer_name] = weighted_sum.reshape(
                        aggregated_model[layer_name].shape
                    )
                    logger.info(
                        f"Layer {layer_name} aggregated using simple average | Weights: {weights_on_samples}"
                    )
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
        from scipy.spatial.distance import cosine

        pairwise_sims = []
        for i in range(N):
            for j in range(i + 1, N):
                sim = cosine(updates_np[i], updates_np[j])
                pairwise_sims.append(sim)
        avg_similarity = np.mean(pairwise_sims) if pairwise_sims else 0.0
        max_similarity = max(pairwise_sims) if pairwise_sims else 0.0
        min_similarity = min(pairwise_sims) if pairwise_sims else 0.0
        logger.info(
            f"Pairwise similarities: avg={avg_similarity:.4f}, max={max_similarity:.4f}, min={min_similarity:.4f}"
        )

        def objective(u):
            """目标函数：最小化加权和的L2范数平方"""
            weighted_sum = np.zeros_like(updates_np[0])
            for i, weight in enumerate(u):
                weighted_sum += weight * updates_np[i]
            return np.linalg.norm(weighted_sum) ** 2

        # 初始权重（均匀分布）
        u0 = np.ones(N) / N
        u0 += np.random.normal(0, 0.05, N)  # 添加小扰动
        u0 = np.clip(u0, 0.05, 0.95)  # 确保初始权重在0-1范围内
        u0 /= np.sum(u0)  # 确保初始权重和为1

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
                options={"maxiter": 500, "ftol": 1e-6, "disp": True},
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
                    for idx in range(n_client):
                        if layer in model_params[idx]:
                            updated_dicts[idx][layer] = (
                                model_params[idx][layer].detach().clone()
                            )
                    continue

                layer_params = [params[layer] for params in model_params]
                original_shapes = [p.shape for p in layer_params]
                # 堆叠为张量（维度：[n_client] + 原始参数维度）
                # 例如：若单客户端参数为(d1, d2)，堆叠后为(n_client, d1, d2)
                W_tensor = torch.stack(layer_params, dim=0)  # 保留原始维度，不展平
                p = W_tensor.ndim  # 张量阶数（p ≥ 2，含客户端维度）
                logger.info(
                    f"Layer {layer} stacked tensor shape: {W_tensor.shape} (p={p})"
                )

                # 5. 计算张量迹范数的次梯度（基于各维度展开矩阵，原文、）
                total_grad = torch.zeros_like(W_tensor)  # 累计各维度次梯度
                # 对每个维度k（除客户端维度外，假设客户端维度为0）计算展开矩阵的迹范数次梯度
                for k in range(0, p - 1):  # 维度从1开始（0为客户端维度）
                    # 沿第k维度展开张量为矩阵
                    # 展开规则：将维度k与客户端维度(0)合并为新的行，其余维度合并为列
                    unfold_dim = k
                    W_unfold = torch.flatten(W_tensor, start_dim=0, end_dim=unfold_dim)
                    W_unfold = torch.flatten(W_unfold, start_dim=1)  # 列：剩余维度合并
                    logger.info(f"Unfolded matrix shape (dim {k}): {W_unfold.shape}")

                    # SVD分解计算迹范数次梯度
                    U, S, Vh = torch.linalg.svd(W_unfold, full_matrices=False)
                    rank = torch.sum(S > 1e-8).item()
                    # 输出rank和奇异值
                    logger.info(
                        f"Unfolded matrix (dim {k}) rank: {rank}, singular values: {S.tolist()}"
                    )
                    if rank == 0:
                        logger.warning(
                            f"Unfolded matrix (dim {k}) has near-zero singular values, skipping"
                        )
                        continue

                    # 有效奇异向量
                    U_eff = U[:, :rank]
                    Vh_eff = Vh[:rank, :]
                    grad_unfold = U_eff @ Vh_eff  # 迹范数次梯度（原文）

                    # 将展开矩阵的梯度还原为张量形状
                    grad_tensor = grad_unfold.reshape(
                        W_tensor.shape
                    )  # 恢复为堆叠张量形状
                    total_grad += grad_tensor  # 累加各维度梯度（默认各维度权重α_k=1）

                # 更新共享层参数（原文：w_i^t ← w_i^t - η_w * ∇L_r）
                W_updated = W_tensor - self.lambda_ * total_grad
                # 输出每行梯度的范数
                logger.info(
                    f"Layer {layer} total gradient norms: {[torch.norm(g, p=2).item() for g in total_grad]}"
                )

                # 7. 还原各客户端参数形状
                for idx in range(n_client):
                    updated_param = W_updated[idx].reshape(original_shapes[idx])
                    updated_dicts[idx][layer] = updated_param

            # 构建返回格式
            personalized = {gid: state for gid, state in zip(client_ids, updated_dicts)}

            return {"model_para_all": personalized}

    def _para_weighted_avg(self, models, recover_fun=None):
        """
        Calculates the weighted average of models.
        """
        training_set_size = 0
        for i in range(len(models)):
            sample_size, _ = models[i]
            training_set_size += sample_size

        sample_size, avg_model = models[0]

        for key in avg_model:
            for i in range(len(models)):
                local_sample_size, local_model = models[i]

                if key not in local_model:
                    continue

                if self.cfg.federate.ignore_weight:
                    weight = 1.0 / len(models)
                elif self.cfg.federate.use_ss:
                    # When using secret sharing, what the server receives
                    # are sample_size * model_para
                    weight = 1.0
                else:
                    weight = local_sample_size / training_set_size

                if not self.cfg.federate.use_ss:
                    local_model[key] = param2tensor(local_model[key])
                if i == 0:
                    avg_model[key] = local_model[key] * weight
                else:
                    avg_model[key] += local_model[key] * weight

            if self.cfg.federate.use_ss and recover_fun:
                avg_model[key] = recover_fun(avg_model[key])
                # When using secret sharing, what the server receives are
                # sample_size * model_para
                avg_model[key] /= training_set_size
                avg_model[key] = torch.FloatTensor(avg_model[key])

        return avg_model
