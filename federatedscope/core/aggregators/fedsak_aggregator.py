import torch
from torch import nn
from federatedscope.core.aggregators import Aggregator
from federatedscope.core.auxiliaries.utils import param2tensor
import logging

logger = logging.getLogger(__name__)


class FedSAKAggregator(Aggregator):
    r"""
    实现论文 FedSAK 的核心算法：
    • 堆叠各客户端共享层 → 施加张量 trace‑norm 正则 → 次梯度更新
    • 返回个性化 slice 给对应客户端
    """

    def __init__(self, config):
        # super().__init__(config, model=model, device=device)
        self.lmbda = config.aggregator.get("lambda_", 1e-3)
        self.lr = config.aggregator.get("lr_shared", 0.05)
        self.share_patterns = config.aggregator.get("filter_keys", [])
        self.cfg = config

        logger.info(
            "FedSAK Aggregator initialized with " + f"lambda={self.lmbda}, lr={self.lr}"
        )
        logger.info(f"Share patterns: {self.share_patterns}")

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

    # ------ 核心聚合函数 ------
    def aggregate(self, agg_info):
        """
        处理来自客户端的模型参数，执行trace-norm正则化
        """
        # 解析聚合信息
        client_feedback = agg_info["client_feedback"]

        if self.lmbda == 0:
            models = []
            for feedback in client_feedback:
                sample_size, model_para = feedback["model_para"]
                models.append((sample_size, model_para))
            # 直接返回平均模型
            avg_model = self._para_weighted_avg(models)
            # 构建返回格式
            personalized = {
                feedback["client_id"]: avg_model for feedback in client_feedback
            }
            return {"model_para_all": personalized}
        else:
            # 从client_feedback中提取必要信息
            client_ids = []
            model_params = []

            for feedback in client_feedback:
                client_ids.append(feedback["client_id"])
                sample_size, model_para = feedback["model_para"]
                # print(type(model_para),type(feedback['model_para']))
                model_params.append(model_para)

            n_client = len(model_params)
            updated_dicts = [dict() for _ in range(n_client)]

            # 处理每个共享模式
            for pattern in self.share_patterns:
                try:
                    # 查找匹配该模式的所有键
                    matching_keys = []
                    for params in model_params:
                        matching_keys.extend([k for k in params.keys() if pattern in k])
                    # 去重
                    matching_keys = list(set(matching_keys))
                    # print(f"Matching keys for
                    # pattern '{pattern}': {matching_keys}")

                    if not matching_keys:
                        logger.warning(f"No keys found matching pattern '{pattern}'")
                        continue

                    # 处理每个匹配的键
                    for k in matching_keys:
                        # 检查所有客户端是否都有此键
                        if all(k in params for params in model_params):
                            # 提取形状信息
                            shapes = [params[k].shape for params in model_params]
                            # 堆叠参数矩阵
                            W = torch.stack(
                                [params[k].flatten() for params in model_params], 0
                            )
                            # 执行SVD分解
                            U, S, Vh = torch.linalg.svd(W, full_matrices=False)

                            # 计算rank
                            # threshold = 1e-4
                            # rank_W = torch.sum(S > \
                            # threshold).item()

                            # 计算最小的奇异值
                            # min_sigma = torch.min(S).item()

                            # 打印日志
                            # logger.info(f"[FedSAK] "
                            #             f"rank={rank_W},
                            #             f"minσ={min_sigma:.2e}, "
                            #             f"Σ1={S[0]:.2e}, Σ2={S[1]:.2e},
                            #             f"Σ3={S[2]:.2e}")

                            # 计算次梯度
                            grad = U @ Vh
                            # 更新参数
                            W_new = W - self.lr * self.lmbda * grad
                            # 还原各客户端参数形状
                            for idx, slice_vec in enumerate(W_new):
                                updated_dicts[idx][k] = slice_vec.reshape(shapes[idx])
                        else:
                            logger.warning(f"Key '{k}' not found in all client models")
                            # 保留原参数
                            for idx in range(n_client):
                                if k in model_params[idx]:
                                    updated_dicts[idx][k] = model_params[idx][k]
                except Exception as e:
                    logger.warning(f"Error processing pattern '{pattern}': {str(e)}")

            # 构建返回格式
            personalized = {cid: state for cid, state in zip(client_ids, updated_dicts)}

            return {"model_para_all": personalized}
