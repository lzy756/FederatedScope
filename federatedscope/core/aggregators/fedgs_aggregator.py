import logging
import copy
import torch
import numpy as np
from scipy.optimize import minimize

from federatedscope.core.aggregators import ClientsAvgAggregator
from federatedscope.core.auxiliaries.utils import param2tensor

logger = logging.getLogger(__name__)


class FedGSAggregator(ClientsAvgAggregator):
    """FedGS 聚合器实现
    
    继承自 ClientsAvgAggregator，实现 FedGS 的分组聚合逻辑
    组内聚合使用加权平均，组间聚合使用FedSAK的trace-norm正则化方法
    """
    def __init__(self, model=None, device='cpu', config=None):
        super().__init__(model, device, config)

        # FedSAK specific parameters
        self.lmbda = config.aggregator.get("lambda_", 1e-3)  # trace-norm正则化强度
        self.lr = config.aggregator.get("lr_shared", 0.05)  # 学习率
        self.share_patterns = config.fedsak.share_patterns  # 共享层配置

    def aggregate_intra_group(self, client_feedback):
        """组内聚合：使用加权平均聚合组内客户端的模型
        
        Args:
            client_feedback: 组内客户端反馈列表，每个元素包含:
                - client_id: 客户端ID
                - sample_size: 样本大小
                - model_para: 模型参数
            
        Returns:
            聚合后的模型参数
        """
        logger.info(
            f'Server: Performing intra-group aggregation with {len(client_feedback)} clients'
        )
        return self._intra_group_aggregate(client_feedback)

    def aggregate_inter_group(self, group_feedback):
        """组间聚合：使用FedSAK的trace-norm正则化方法聚合所有组的模型
        
        Args:
            group_feedback: 所有组的反馈列表，每个元素包含:
                - client_id: 组ID
                - sample_size: 组内总样本大小
                - model_para: 组内聚合后的模型参数
            
        Returns:
            聚合后的个性化模型参数
        """
        logger.info(
            f'Server: Performing inter-group aggregation with {len(group_feedback)} groups using FedSAK'
        )
        return self._inter_group_aggregate_fedsak(group_feedback)

    def aggregate(self, agg_info):
        """执行分组聚合（保留此方法以保持向后兼容）
        
        Args:
            agg_info: 包含客户端反馈的信息字典
            
        Returns:
            聚合后的模型参数
        """
        client_feedback = agg_info["client_feedback"]

        # 检查是否是组间聚合（通过检查 client_id 是否为字符串类型）
        if client_feedback and isinstance(client_feedback[0]['client_id'],
                                          str):
            return self.aggregate_inter_group(client_feedback)
        else:
            return {'model': self.aggregate_intra_group(client_feedback)}

    def _intra_group_aggregate(self, models):
        """组内聚合：使用共识最大化聚合组内客户端的模型

        Args:
            models: 组内客户端模型列表

        Returns:
            聚合后的模型参数
        """
        if len(models) == 1:
            # 如果只有一个客户端，直接返回其模型
            return models[0]['model_para']

        logger.info(f"Performing consensus maximization aggregation with {len(models)} clients")

        # 获取第一个模型作为基准
        aggregated_model = copy.deepcopy(models[0]['model_para'])

        # 将所有客户端的模型参数展平为单个向量
        flattened_models = []
        layer_shapes = {}
        layer_order = list(aggregated_model.keys())
        
        for model in models:
            flattened_params = []
            for layer_name in layer_order:
                if layer_name in model['model_para']:
                    layer_param = model['model_para'][layer_name]
                    if layer_name not in layer_shapes:
                        layer_shapes[layer_name] = layer_param.shape
                    flattened_params.append(layer_param.flatten())
                else:
                    logger.warning(f"Layer {layer_name} not found in client model, using zeros")
                    if layer_name in layer_shapes:
                        zero_param = torch.zeros(layer_shapes[layer_name]).flatten()
                        flattened_params.append(zero_param)
            
            # 将所有层的参数连接成一个向量
            full_model_vector = torch.cat(flattened_params)
            flattened_models.append(full_model_vector)

        # 对整个模型向量执行共识最大化聚合
        aggregated_vector = self._consensus_maximization_vector(flattened_models)
        
        # 将聚合后的向量拆分回各层
        start_idx = 0
        for layer_name in layer_order:
            if layer_name in layer_shapes:
                layer_shape = layer_shapes[layer_name]
                layer_size = torch.prod(torch.tensor(layer_shape)).item()
                layer_vector = aggregated_vector[start_idx:start_idx + layer_size]
                aggregated_model[layer_name] = layer_vector.reshape(layer_shape)
                start_idx += layer_size

        return aggregated_model

    def _consensus_maximization_vector(self, model_vectors):
        """对整个模型向量执行共识最大化聚合

        Args:
            model_vectors: 所有客户端的模型向量列表

        Returns:
            聚合后的模型向量
        """
        N = len(model_vectors)
        if N == 1:
            return model_vectors[0]

        # 计算每个客户端模型向量的L2范数
        norms = []
        normalized_vectors = []

        for vector in model_vectors:
            norm = torch.norm(vector, p=2)
            norms.append(norm)
            if norm > 1e-8:  # 避免除零
                normalized_vectors.append(vector / norm)
            else:
                normalized_vectors.append(torch.zeros_like(vector))

        # 计算平均范数
        avg_norm = sum(norms) / N

        # 求解优化问题：argmin_u ||∑_k u_k * d_{u,k}||_2^2
        weights = self._solve_consensus_weights(normalized_vectors)

        # 计算最终的聚合向量
        aggregated_vector = torch.zeros_like(model_vectors[0])
        for i, (weight, norm_vector) in enumerate(zip(weights, normalized_vectors)):
            aggregated_vector += weight * norm_vector

        # 乘以平均范数
        final_vector = avg_norm * aggregated_vector
        return final_vector

    def _solve_consensus_weights(self, normalized_updates):
        """求解共识权重优化问题

        Args:
            normalized_updates: 归一化后的更新向量列表

        Returns:
            优化后的权重向量
        """
        N = len(normalized_updates)
        if N == 1:
            return [1.0]

        # 将张量转换为numpy数组进行优化
        updates_np = [update.detach().cpu().numpy() for update in normalized_updates]

        def objective(u):
            """目标函数：||∑_k u_k * d_{u,k}^(l)||_2^2"""
            weighted_sum = np.zeros_like(updates_np[0])
            for i, weight in enumerate(u):
                weighted_sum += weight * updates_np[i]
            return np.linalg.norm(weighted_sum) ** 2

        # 初始权重（均匀分布）
        u0 = np.ones(N) / N

        # 约束条件：权重和为1，且每个权重非负
        constraints = [
            {'type': 'eq', 'fun': lambda u: np.sum(u) - 1.0}
        ]
        bounds = [(0, 1) for _ in range(N)]

        try:
            # 使用scipy优化求解
            result = minimize(objective, u0, method='SLSQP',
                            bounds=bounds, constraints=constraints,
                            options={'maxiter': 1000, 'ftol': 1e-9})

            if result.success:
                weights = result.x.tolist()
                logger.debug(f"Consensus optimization converged with weights: {weights}")
            else:
                logger.warning(f"Consensus optimization failed: {result.message}, using uniform weights")
                weights = [1.0/N] * N

        except Exception as e:
            logger.warning(f"Error in consensus optimization: {str(e)}, using uniform weights")
            weights = [1.0/N] * N

        return weights

    def _weighted_average_layer(self, models, layer_name):
        """对单层参数执行加权平均（备用方法）

        Args:
            models: 客户端模型列表
            layer_name: 层名称

        Returns:
            加权平均后的层参数
        """
        total_samples = sum(
            model['sample_size'] if isinstance(model['sample_size'], int)
            else int(model['sample_size']) for model in models
        )

        weighted_layer = None
        for model in models:
            if layer_name in model['model_para']:
                weight = (model['sample_size'] if isinstance(model['sample_size'], int)
                         else int(model['sample_size'])) / total_samples

                if weighted_layer is None:
                    weighted_layer = model['model_para'][layer_name] * weight
                else:
                    weighted_layer += model['model_para'][layer_name] * weight

        return weighted_layer if weighted_layer is not None else torch.zeros_like(models[0]['model_para'][layer_name])

    def _inter_group_aggregate_fedsak(self, group_models):
        """组间聚合：使用FedSAK的trace-norm正则化方法聚合所有组的模型
        
        Args:
            group_models: 所有组的模型列表
            
        Returns:
            聚合后的个性化模型参数
        """
        if self.lmbda == 0:
            # 如果lambda为0，直接使用加权平均
            models = []
            for feedback in group_models:
                sample_size, model_para = feedback['sample_size'], feedback[
                    'model_para']
                models.append((sample_size, model_para))
            # 直接返回平均模型
            avg_model = self._para_weighted_avg(models)
            # 构建返回格式
            personalized = {
                feedback['client_id']: avg_model
                for feedback in group_models
            }
            return {"model_para_all": personalized}
        else:
            # 从group_models中提取必要信息
            group_ids = []
            model_params = []

            for feedback in group_models:
                group_ids.append(feedback['client_id'])
                model_para = feedback['model_para']
                model_params.append(model_para)

            n_groups = len(model_params)
            updated_dicts = [dict() for _ in range(n_groups)]

            # 处理每个共享模式
            for pattern in self.share_patterns:
                try:
                    # 查找匹配该模式的所有键
                    matching_keys = []
                    for params in model_params:
                        matching_keys.extend(
                            [k for k in params.keys() if pattern in k])
                    # 去重
                    matching_keys = list(set(matching_keys))

                    if not matching_keys:
                        logger.warning(
                            f"No keys found matching pattern '{pattern}'")
                        continue

                    # 处理每个匹配的键
                    for k in matching_keys:
                        # 检查所有组是否都有此键
                        if all(k in params for params in model_params):
                            # 提取形状信息
                            shapes = [
                                params[k].shape for params in model_params
                            ]
                            # 堆叠参数矩阵
                            W = torch.stack([
                                params[k].flatten() for params in model_params
                            ], 0)
                            # 执行SVD分解
                            U, S, Vh = torch.linalg.svd(W, full_matrices=False)

                            # 计算rank
                            threshold = 1e-4
                            rank_W = torch.sum(S > \
                            threshold).item()

                            # 计算最小的奇异值
                            min_sigma = torch.min(S).item()

                            # # 打印日志
                            # logger.info(
                            #     f"[FedSAK] rank={rank_W}, minσ={min_sigma:.2e}, Σ1={S[0]:.2e}, Σ2={S[1]:.2e}, Σ3={S[2]:.2e}"
                            # )

                            # 计算次梯度
                            grad = U @ Vh
                            # 更新参数
                            W_new = W - self.lr * self.lmbda * grad
                            # 还原各组参数形状
                            for idx, slice_vec in enumerate(W_new):
                                updated_dicts[idx][k] = slice_vec.reshape(
                                    shapes[idx])
                        else:
                            logger.warning(
                                f"Key '{k}' not found in all group models")
                            # 保留原参数
                            for idx in range(n_groups):
                                if k in model_params[idx]:
                                    updated_dicts[idx][k] = model_params[idx][
                                        k]
                except Exception as e:
                    logger.warning(
                        f"Error processing pattern '{pattern}': {str(e)}")

            # 构建返回格式
            personalized = {
                gid: state
                for gid, state in zip(group_ids, updated_dicts)
            }

            return {"model_para_all": personalized}

    def _para_weighted_avg(self, models, recover_fun=None):
        """计算模型参数的加权平均
        
        Args:
            models: 模型列表，每个元素为(sample_size, model_para)元组
            recover_fun: 恢复函数（用于secret sharing）
            
        Returns:
            加权平均后的模型参数
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

                if self._cfg.federate.ignore_weight:
                    weight = 1.0 / len(models)
                elif self._cfg.federate.use_ss:
                    # When using secret sharing, what the server receives
                    # are sample_size * model_para
                    weight = 1.0
                else:
                    weight = local_sample_size / training_set_size

                if not self._cfg.federate.use_ss:
                    local_model[key] = param2tensor(local_model[key])
                if i == 0:
                    avg_model[key] = local_model[key] * weight
                else:
                    avg_model[key] += local_model[key] * weight

            if self._cfg.federate.use_ss and recover_fun:
                avg_model[key] = recover_fun(avg_model[key])
                # When using secret sharing, what the server receives are
                # sample_size * model_para
                avg_model[key] /= training_set_size
                avg_model[key] = torch.FloatTensor(avg_model[key])

        return avg_model
