"""
FedLSA (联邦可学习语义锚点) 配置

本文件定义FedLSA方法的所有超参数配置项
"""
from federatedscope.core.configs.config import CN
from federatedscope.register import register_config


def extend_fedlsa_cfg(cfg):
    """
    添加FedLSA特定的配置选项

    配置组织:
    1. 客户端参数: 控制本地训练的超球面对比学习
    2. 服务器参数: 控制语义锚点的学习
    3. 模型参数: 网络架构配置

    Args:
        cfg: 全局配置对象
    """
    # ---------------------------------------------------------------------- #
    # FedLSA配置
    # ---------------------------------------------------------------------- #
    cfg.fedlsa = CN()

    # 是否使用FedLSA方法
    cfg.fedlsa.use = False

    # ========== 客户端超参数 ==========
    # 紧凑性损失的权衡参数 (论文中的λ, 公式10)
    # L_HC = L_CE + λ·L_COM
    # 推荐范围: {0.1, 0.7}
    cfg.fedlsa.lambda_com = 0.5

    # 对比学习的温度参数 (论文中的τ, 公式8)
    # τ 控制分布的尖锐程度: 较小的τ使分布更尖锐
    # 推荐范围: {0.1, 0.2}
    cfg.fedlsa.tau = 0.1

    # 是否使用超球面投影器
    # True: 使用投影器将嵌入z投影到超球面h
    # False: 直接使用嵌入z (不推荐)
    cfg.fedlsa.use_projector = True

    # 投影器输入维度 (特征嵌入维度，论文中的I)
    # 应与模型的嵌入维度一致
    cfg.fedlsa.projector_input_dim = 512

    # 投影器输出维度 (超球面嵌入维度，论文中的L)
    # 通常小于输入维度以降低计算复杂度
    cfg.fedlsa.projector_output_dim = 128

    # 是否在客户端间共享投影器参数
    # True: 投影器参与联邦平均
    # False: 每个客户端独立的投影器
    cfg.fedlsa.share_projector = True

    # ========== 服务器超参数 ==========
    # 分离损失的权衡参数 (论文中的α, 公式5)
    # L_LSA = L_ACE + α·L_SEP
    # 推荐范围: {0.02, 0.4}
    cfg.fedlsa.alpha_sep = 0.4

    # 服务器端训练语义锚点的轮数 (论文中的E_s)
    # 每个通信轮次开始前，服务器训练锚点E_s轮
    cfg.fedlsa.anchor_train_epochs = 500

    # 锚点优化的学习率 (论文中的η)
    cfg.fedlsa.anchor_lr = 0.001

    # ---------------------------------------------------------------------- #
    # FedLSA的模型配置
    # ---------------------------------------------------------------------- #
    # 类别数量 (论文中的C，用于锚点初始化)
    # Note: This is handled by the model config, we just document it here
    # cfg.model.num_classes should be set by the model configuration

    # ---------------------------------------------------------------------- #
    # Trainer和Worker配置
    # ---------------------------------------------------------------------- #
    # 这些将由用户设置，或在fedlsa.use=True时自动设置


# 注册配置扩展函数到FederatedScope框架
register_config("fedlsa", extend_fedlsa_cfg)

