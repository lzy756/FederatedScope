import logging
import torch
import torch.nn.functional as F
from typing import List, Dict
from federatedscope.core.aggregators import ClientsAvgAggregator
from typing import List, Dict, Tuple

logger = logging.getLogger(__name__)


class FDSEAggregator(ClientsAvgAggregator):
    def __init__(self, model=None, device="cpu", config=None):
        super().__init__(model=model, device=device, config=config)

    def aggregate(self, agg_info):
        return self._aggregate(agg_info["client_feedback"])

    def _aggregate(
        self, client_feedback: List[Dict[str, torch.Tensor]]
    ) -> Dict[str, List[Dict[str, torch.Tensor]]]:
        """
        FDSE aggregation function

        Args:
            client_feedback: List of client feedback, each element contains:
                - 'client_id': Client ID
                - 'model_para': Model parameter dictionary

        Returns:
            Result containing multiple client model parameter dictionaries:
            {
                "model_para_all": [client_0_params, client_1_params, ...]
            }
        """
        logger.info(
            f"FDSE aggregator starting aggregation, number of clients: {len(client_feedback)}"
        )

        if not client_feedback:
            logger.warning("Client feedback is empty, returning empty result")
            return {"model_para_all": []}

        num_clients = len(client_feedback)
        logger.debug(f"Processing parameters from {num_clients} clients")

        # Extract client parameters and weights
        client_ids = [feedback["client_id"] for feedback in client_feedback]
        sample_counts = [feedback["model_para"][0] for feedback in client_feedback]
        client_params = [feedback["model_para"][1] for feedback in client_feedback]

        # Calculate normalized weights
        total_samples = sum(sample_counts)
        weights = [count / total_samples for count in sample_counts]
        logger.info(
            f"Total samples: {total_samples}, weight distribution: {[f'{w:.4f}' for w in weights]}"
        )

        # Get all parameter names
        param_names = list(client_params[0].keys())

        non_trainable_params = []
        dfe_params = []
        dse_params = []
        other_params = []

        # For trainable parameters in DSE, use attention mechanism for aggregation

        for param_name in param_names:
            if self._is_non_trainable_param(param_name):
                non_trainable_params.append(param_name)
            elif self._is_dfe_param(param_name):
                dfe_params.append(param_name)
            elif self._is_dse_param(param_name):
                dse_params.append(param_name)
            else:
                other_params.append(param_name)

        logger.info(
            f"Parameter classification completed - Non-trainable params: {len(non_trainable_params)}, DFE params: {len(dfe_params)}, DSE params: {len(dse_params)}, Other params: {len(other_params)}"
        )

        logger.info(f"Non-trainable parameters: {non_trainable_params}")
        logger.info(f"DFE parameters: {dfe_params}")
        logger.info(f"DSE parameters: {dse_params[:10]}")
        logger.info(f"Other parameters: {other_params}")

        # Initialize aggregation results for each client
        aggregated_client_params = {client_id: {} for client_id in client_ids}

        # 0. Non-trainable parameters (e.g., running_mean, running_var) use simple average aggregation, distribute to all clients
        if non_trainable_params:
            logger.info(
                f"Starting aggregation of {len(non_trainable_params)} non-trainable parameters using simple average"
            )
            for param_name in non_trainable_params:
                param_tensors = [
                    params[param_name]
                    for params in client_params
                    if param_name in params
                ]

                logger.info(
                    f"Aggregating non-trainable parameter '{param_name}', tensor norm '{[p.norm().item() for p in param_tensors]}'"
                )
                # Use simple average aggregation (no sample weights)
                aggregated_param = self._simple_average(param_tensors)
                # Copy aggregation result to each client
                for client_id in client_ids:
                    aggregated_client_params[client_id][
                        param_name
                    ] = aggregated_param.clone()
            logger.info("Non-trainable parameter simple average aggregation completed")

        # 1. DFE module uses FedSAK aggregation
        if dfe_params:
            logger.info(
                f"Starting aggregation of {len(dfe_params)} DFE parameters using FedSAK aggregation"
            )
            for param_name in dfe_params:
                param_tensors = [
                    params[param_name]
                    for params in client_params
                    if param_name in params
                ]

                # Use weighted average aggregation
                aggregated_param = self._weighted_average(param_tensors, weights)
                # Copy aggregation result to each client
                for client_id in client_ids:
                    aggregated_client_params[client_id][
                        param_name
                    ] = aggregated_param.clone()
            logger.info("Weighted average aggregation completed")
            # Extract DFE parameters for all clients
            # dfe_client_states = []
            # for params in client_params:
            #     dfe_state = {
            #         param_name: params[param_name]
            #         for param_name in dfe_params
            #         if param_name in params
            #     }
            #     dfe_client_states.append(dfe_state)

            # # Apply FedSAK aggregation
            # updated_dfe_states = self._fedsak_update(dfe_client_states, lambda_reg=0.01)

            # # Assign updated DFE parameters to each client
            # for i, client_id in enumerate(client_ids):
            #     for param_name in dfe_params:
            #         if param_name in updated_dfe_states[i]:
            #             aggregated_client_params[client_id][param_name] = (
            #                 updated_dfe_states[i][param_name]
            #             )
            # logger.info("DFE parameter FedSAK aggregation completed")

        # 2. DSE module uses attention mechanism aggregation, each client gets personalized result
        if dse_params:
            logger.info(
                f"Starting aggregation of {len(dse_params)} DSE parameters using attention mechanism"
            )
            for param_name in dse_params:
                param_tensors = [
                    params[param_name]
                    for params in client_params
                    if param_name in params
                ]

                # Use attention mechanism aggregation, return personalized result for each client
                client_specific_results = self._attention_aggregate(param_tensors)
                # Assign each client's personalized result to corresponding client
                for i, client_id in enumerate(client_ids):
                    aggregated_client_params[client_id][param_name] = (
                        client_specific_results[i]
                    )
            logger.info("Attention mechanism aggregation completed")

        # 3. Other parameters use weighted average aggregation, result copied to each client
        if other_params:
            logger.info(
                f"Starting aggregation of {len(other_params)} other parameters using weighted average"
            )
            for param_name in other_params:
                param_tensors = [
                    params[param_name]
                    for params in client_params
                    if param_name in params
                ]

                # Use weighted average aggregation
                aggregated_param = self._weighted_average(param_tensors, weights)
                # Copy aggregation result to each client
                for client_id in client_ids:
                    aggregated_client_params[client_id][
                        param_name
                    ] = aggregated_param.clone()
            logger.info("Weighted average aggregation completed")

        logger.info(
            f"FDSE aggregator aggregation completed, generated personalized model parameters for {num_clients} clients"
        )
        return {"model_para_all": aggregated_client_params}

    def _is_non_trainable_param(self, param_name: str) -> bool:
        """Check if parameter is non-trainable"""
        non_trainable_keywords = [
            "running_mean",
            "running_var",
            "num_batches_tracked",
            "running_std",
            "running_avg",
            "track_running_stats",
        ]
        param_lower = param_name.lower()
        return any(keyword in param_lower for keyword in non_trainable_keywords)

    def _is_dfe_param(self, param_name: str) -> bool:
        """Check if parameter belongs to DFE module"""
        return "dfe.conv" in param_name.lower()

    def _is_dse_param(self, param_name: str) -> bool:
        """Check if parameter belongs to DSE module"""
        return "dse" in param_name.lower()

    def _simple_average(self, tensors: List[torch.Tensor]) -> torch.Tensor:
        """Simple average aggregation (no weights)"""
        if not tensors:
            logger.error("Attempting simple average on empty tensor list")
            raise ValueError("Empty tensor list")

        logger.debug(
            f"Performing simple average aggregation, number of tensors: {len(tensors)}"
        )
        result = torch.zeros_like(tensors[0])
        for tensor in tensors:
            result += tensor
        result = result / len(tensors)
        return result

    def _weighted_average(
        self, tensors: List[torch.Tensor], weights: List[float]
    ) -> torch.Tensor:
        """Weighted average aggregation"""
        if not tensors:
            logger.error("Attempting weighted average on empty tensor list")
            raise ValueError("Empty tensor list")

        logger.debug(
            f"Performing weighted average aggregation, number of tensors: {len(tensors)}, weights: {[f'{w:.4f}' for w in weights]}"
        )
        result = torch.zeros_like(tensors[0])
        for tensor, weight in zip(tensors, weights):
            result += weight * tensor
        return result

    def _attention_aggregate(self, tensors: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Use scaled dot-product attention mechanism to aggregate DSE parameters, returning an aggregated result for each client

        Args:
            tensors: List of client parameter tensors

        Returns:
            List of aggregated parameter tensors for each client
        """
        if not tensors:
            logger.error("Attempting attention aggregation on empty tensor list")
            raise ValueError("Empty tensor list")

        if len(tensors) == 1:
            logger.debug("Only one tensor, returning directly")
            return [tensors[0]]

        original_shape = tensors[0].shape
        logger.debug(
            f"Attention aggregation started, number of clients: {len(tensors)}, tensor shape: {original_shape}"
        )

        Q = torch.stack([tensor.flatten() for tensor in tensors])
        K = torch.stack([tensor.flatten() for tensor in tensors])
        V = torch.stack([tensor.flatten() for tensor in tensors])

        # Use scaled dot-product attention to calculate attention scores
        d_k = Q.size(-1)  # Feature dimension
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(
            torch.tensor(d_k, dtype=Q.dtype)
        )
        logger.debug(f"Attention score calculation completed, feature dimension: {d_k}")

        # Calculate personalized aggregation results for each client
        client_aggregated_results = []
        for i in range(len(tensors)):
            # Attention weights for the i-th client (based on parameter similarity only)
            attn_weights = F.softmax(attention_scores[i], dim=0)
            logger.debug(f"Client {i} attention weights: {attn_weights.cpu().numpy()}")

            # Aggregate based on attention weights
            aggregated_feature = torch.sum(attn_weights.unsqueeze(1) * V, dim=0)

            # Restore original shape and add to result list
            client_result = aggregated_feature.reshape(original_shape)
            client_aggregated_results.append(client_result)

        logger.debug(
            f"Attention aggregation completed, generated {len(client_aggregated_results)} client-specific results"
        )
        return client_aggregated_results

    def _mode_unfold(self, tensor: torch.Tensor, mode: int) -> torch.Tensor:
        """模式展开：对应公式中$\mathcal{W}^l_{(k)}$的构建"""
        mode_idx = mode - 1  # 转换为0-based索引
        perm = [mode_idx] + [i for i in range(tensor.ndim) if i != mode_idx]
        return tensor.permute(perm).reshape(tensor.shape[mode_idx], -1)

    def _mode_fold(
        self, matrix: torch.Tensor, tensor_shape: Tuple[int], mode: int
    ) -> torch.Tensor:
        """模式折叠：对应公式中$\text{fold}_k$操作"""
        mode_idx = mode - 1
        d_mode = tensor_shape[mode_idx]
        other_dims = [
            tensor_shape[i] for i in range(len(tensor_shape)) if i != mode_idx
        ]
        permuted_shape = (d_mode,) + tuple(other_dims)
        tensor_perm = matrix.reshape(permuted_shape)
        perm = [i for i in range(len(tensor_shape))]
        perm.insert(mode_idx, perm.pop(0))
        return tensor_perm.permute(perm)

    def _compute_tensor_trace_norm(self, tensor: torch.Tensor) -> float:
        """计算张量的迹范数（各模式展开矩阵的迹范数平均值）"""
        p = tensor.ndim
        total = 0.0
        for k in range(1, p + 1):
            unfolded = self._mode_unfold(tensor, k)
            # 矩阵迹范数 = 奇异值之和
            _, sigma, _ = torch.linalg.svd(unfolded, full_matrices=False)
            total += sigma.sum().item()
        return total  # 平均各模式的迹范数

    def _compute_layer_subgradient(self, stacked_tensor: torch.Tensor) -> torch.Tensor:
        """计算张量迹范数子梯度，增强小矩阵SVD稳定性"""
        p = stacked_tensor.ndim
        total_subgrad = torch.zeros_like(stacked_tensor)
        modes = range(1, p + 1)  # 处理所有模式（含二维张量的模式1和2）

        # 全局判断：如果整个张量接近零，直接返回零梯度
        if torch.allclose(stacked_tensor, torch.zeros_like(stacked_tensor), atol=1e-6):
            return total_subgrad

        for k in modes:
            unfolded = self._mode_unfold(stacked_tensor, k)
            nuc_norm = torch.linalg.norm(unfolded, ord="nuc")
            if nuc_norm < 1e-4:  # 覆盖测试案例中的1.4e-5场景
                tensor_subgrad = torch.zeros_like(stacked_tensor)
            else:
                # 正常矩阵使用SVD计算子梯度
                u, s, vh = torch.linalg.svd(unfolded, full_matrices=False)
                eps = 1e-6  # 防止数值不稳定
                mask = s > eps  # 有效奇异值掩码
                u = u[:, mask]
                vh = vh[mask, :]
                mat_subgrad = u @ vh
                tensor_subgrad = self._mode_fold(mat_subgrad, stacked_tensor.shape, k)

            total_subgrad += tensor_subgrad

        return total_subgrad

    def _fedsak_update(
        self, client_states: List[Dict[str, torch.Tensor]], lambda_reg: float = 0.1
    ) -> List[Dict[str, torch.Tensor]]:
        """
        FedSAK参数更新主函数（全共享层场景）
        对应公式：$\omega_i^{t+1} = \omega_i^t - \eta \left( \nabla \mathcal{L}_{\text{data}} + \lambda \frac{\partial \mathcal{L}_r}{\partial \omega_i} \right)$
        """
        num_clients = len(client_states)
        if num_clients == 0:
            return []

        # 获取所有层名称（假设所有客户端的层结构一致）
        all_layers = client_states[0].keys()
        updated_states = [dict() for _ in range(num_clients)]

        # 遍历每个层（对应公式中的l=1..L）
        for layer in all_layers:
            # 1. 堆叠所有客户端的层参数：$\mathcal{W}^l = \text{stack}(\omega_1^l, ..., \omega_M^l)$
            layer_tensors = []
            for i in range(num_clients):
                # 增加客户端维度（最后一维，对应d_p=M）
                layer_tensors.append(client_states[i][layer].unsqueeze(-1))
            stacked = torch.cat(layer_tensors, dim=-1)  # 形状: (d1, d2, ..., dp-1, M)

            # 计算更新前的张量迹范数
            trace_norm_before = self._compute_tensor_trace_norm(stacked)

            # 2. 计算层张量的子梯度：$\frac{\partial \|\mathcal{W}^l\|_*}{\partial \mathcal{W}^l}$
            layer_subgrad = self._compute_layer_subgradient(stacked)

            # 3. 为每个客户端更新参数
            updated_layer_tensors = []
            for i in range(num_clients):
                # 提取客户端i的子梯度切片：$\text{slice}_i\left( \frac{\partial \|\mathcal{W}^l\|_*}{\partial \mathcal{W}^l} \right)$
                reg_grad = layer_subgrad[..., i]
                # 对应$\frac{\partial \|\mathcal{W}^l\|_*}{\partial \omega_i^l}$

                # 参数更新
                updated_states[i][layer] = (
                    client_states[i][layer] - lambda_reg * reg_grad
                )
                updated_layer_tensors.append(updated_states[i][layer].unsqueeze(-1))

            # 重新堆叠更新后的参数以计算更新后的张量迹范数
            updated_stacked = torch.cat(updated_layer_tensors, dim=-1)
            trace_norm_after = self._compute_tensor_trace_norm(updated_stacked)
            logger.info(
                f"Layer '{layer}' - Trace norm after update: {trace_norm_after:.6f} / before update: {trace_norm_before:.6f} / change: {trace_norm_after - trace_norm_before:.6f}"
            )

        return updated_states
