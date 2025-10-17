"""
FedLSA (联邦可学习语义锚点) Trainer实现

参考论文:
Beyond Federated Prototype Learning: Learnable Semantic Anchors with
Hyperspherical Contrast for Domain-Skewed Data (AAAI 2025)

核心功能:
1. 超球面投影: 将特征嵌入投影到单位超球面
2. 紧凑性损失: 基于von Mises-Fisher分布的对比学习
3. 语义锚点引导: 使用服务器学习的语义锚点进行监督
"""
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from federatedscope.core.trainers.torch_trainer import GeneralTorchTrainer
from federatedscope.core.trainers.enums import MODE, LIFECYCLE
from federatedscope.core.trainers.context import CtxVar

logger = logging.getLogger(__name__)


class FedLSATrainer(GeneralTorchTrainer):
    """
    FedLSA方法的训练器，用于处理域偏斜数据

    主要特性:
    - 超球面投影: 确保嵌入在单位超球面上均匀分布
    - 对比损失: 基于von Mises-Fisher分布的类内紧凑性学习
    - 语义锚点: 接收服务器学习的语义锚点进行监督学习
    """

    def __init__(self,
                 model,
                 data,
                 device,
                 config,
                 only_for_eval=False,
                 monitor=None):
        super(FedLSATrainer, self).__init__(
            model, data, device, config, only_for_eval, monitor)

        # FedLSA特定参数
        self.lambda_com = config.fedlsa.lambda_com  # λ: 紧凑性损失权重 (公式10)
        self.tau = config.fedlsa.tau  # τ: 温度参数，控制对比学习的集中度 (公式8)

        # 语义锚点 (从服务器接收)
        self.semantic_anchors = None
        self.num_classes = config.model.num_classes

        # 如需要，构建超球面投影器
        if config.fedlsa.use_projector:
            self._build_hyperspherical_projector()

        logger.info(f"FedLSATrainer初始化完成: lambda_com={self.lambda_com}, tau={self.tau}")

    def _build_hyperspherical_projector(self):
        """
        构建超球面投影器网络，将嵌入映射到超球面空间

        架构: 2层MLP (Multi-Layer Perceptron)
        - 输入维度: 编码器输出维度 (I)
        - 输出维度: 超球面嵌入维度 (L)

        作用: 将欧几里得空间的嵌入 z 投影到超球面空间 h
              h = norm(projector(z))，其中 ||h||_2 = 1
        """
        # 从模型获取特征维度
        if hasattr(self.ctx.model, 'out_channels'):
            in_dim = self.ctx.model.out_channels
        elif hasattr(self.ctx.model, 'feature_dim'):
            in_dim = self.ctx.model.feature_dim
        else:
            in_dim = self.cfg.fedlsa.projector_input_dim

        out_dim = self.cfg.fedlsa.projector_output_dim

        # 构建2层MLP投影器
        self.projector = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.ReLU(),
            nn.Linear(in_dim, out_dim)
        ).to(self.ctx.device)

        logger.info(f"已构建超球面投影器: {in_dim} -> {out_dim}")

    def update_semantic_anchors(self, anchors):
        """
        更新从服务器接收的语义锚点

        语义锚点是服务器端学习的类别表示，独立于具体域，
        用于指导客户端的超球面对比学习

        Args:
            anchors: 语义锚点张量，形状 [C, D]
                    C: 类别数量
                    D: 超球面嵌入维度
        """
        self.semantic_anchors = anchors.to(self.ctx.device)
        logger.debug(f"已更新语义锚点，形状: {anchors.shape}")

    def normalize_embedding(self, embeddings):
        """
        将嵌入归一化到单位超球面

        数学表达: h = z / ||z||_2

        超球面空间的优势:
        1. 确保嵌入均匀分布，避免特征坍塌
        2. 使距离度量更有意义 (角度距离)
        3. 适合von Mises-Fisher分布建模

        Args:
            embeddings: 嵌入张量，形状 [B, D]
                       B: batch size
                       D: 嵌入维度

        Returns:
            归一化后的嵌入，满足 ||h||_2 = 1
        """
        return F.normalize(embeddings, p=2, dim=1)

    def compute_compactness_loss(self, h_embeddings, labels):
        """
        计算紧凑性损失 (L_COM)，基于von Mises-Fisher分布

        论文公式8:
        L_COM = -log( exp(a_{y_i}^T h_i / τ) / Σ_j exp(a_j^T h_i / τ) )

        物理意义:
        - 将每个超球面嵌入 h_i 拉向其对应的语义锚点 a_{y_i}
        - 实现类内紧凑性 (intra-class compactness)
        - 基于von Mises-Fisher分布的最大似然估计

        实现细节:
        1. 计算嵌入与所有锚点的相似度 (内积)
        2. 温度缩放: sim / τ，控制分布的尖锐程度
        3. 使用交叉熵损失 (等价于负对数似然)

        Args:
            h_embeddings: 归一化的超球面嵌入 [B, D]
                         B: batch size
                         D: 超球面嵌入维度
            labels: 真实标签 [B]

        Returns:
            紧凑性损失标量
        """
        if self.semantic_anchors is None:
            # 注意: 第0轮初始化时，客户端可能还未收到锚点，这是正常的
            # 此时跳过紧凑性损失，仅使用分类损失 L_CE 进行训练
            logger.debug("语义锚点尚未接收，跳过紧凑性损失计算 (这在第0轮初始化时是正常的)")
            return torch.tensor(0.0, device=h_embeddings.device)

        # 计算嵌入与所有锚点的相似度
        # h_embeddings: [B, D], semantic_anchors: [C, D]
        # 相似度 = h · a^T (都已归一化，相当于余弦相似度)
        # logits: [B, C]
        logits = torch.matmul(h_embeddings, self.semantic_anchors.T) / self.tau

        # 计算负对数似然 (等价于交叉熵)
        # 这会将每个样本拉向其对应类别的锚点
        loss = F.cross_entropy(logits, labels)

        return loss

    def _forward_with_embedding(self, model, x):
        """
        前向传播并提取最终分类器之前的嵌入

        支持的模型架构:
        1. FedLSACNN: conv1 → conv2 → fc1(嵌入z) → fc2 → fc3(预测q)
        2. ConvNet2:  conv1 → conv2 → fc1(嵌入z) → fc2(预测q)

        提取嵌入的目的:
        - 嵌入z用于投影到超球面空间: h = norm(projector(z))
        - 分离特征学习(z)和分类(q)，使超球面学习更有效

        Args:
            model: 神经网络模型
            x: 输入张量 [B, C, H, W]

        Returns:
            pred: 模型预测(logits) [B, num_classes]
            z: 特征嵌入 [B, hidden_dim]
        """
        # For CNN models, we need to manually extract features
        # by forwarding through all layers except the final classifier

        # 检测是否有fc3 (FedLSACNN的3层全连接架构)
        if hasattr(model, 'fc3') and hasattr(model, 'fc2') and hasattr(model, 'fc1'):
            # FedLSACNN架构: 卷积层 -> fc1 -> fc2 -> fc3
            z = x

            # 卷积层处理
            if hasattr(model, 'conv1'):
                z = model.conv1(z)
                if hasattr(model, 'bn1') and (model.use_bn if hasattr(model, 'use_bn') else False):
                    z = model.bn1(z)
                z = model.relu(z)
                z = model.maxpool(z)

            if hasattr(model, 'conv2'):
                z = model.conv2(z)
                if hasattr(model, 'bn2') and (model.use_bn if hasattr(model, 'use_bn') else False):
                    z = model.bn2(z)
                z = model.relu(z)
                z = model.maxpool(z)

            # 展平
            z = z.view(z.size(0), -1)

            # 应用dropout（如果存在）
            if hasattr(model, 'dropout'):
                z = F.dropout(z, p=model.dropout, training=model.training)

            # FC层1 (这是我们需要的嵌入z)
            embedding = model.relu(model.fc1(z))

            # 继续前向传播通过fc2和fc3得到预测
            if hasattr(model, 'dropout'):
                z = F.dropout(embedding, p=model.dropout, training=model.training)
            else:
                z = embedding
            z = model.relu(model.fc2(z))

            if hasattr(model, 'dropout'):
                z = F.dropout(z, p=model.dropout, training=model.training)
            pred = model.fc3(z)

            return pred, embedding

        # 检测是否有fc2和fc1 (ConvNet2的2层全连接架构)
        elif hasattr(model, 'fc2') and hasattr(model, 'fc1'):
            # ConvNet2架构: 卷积层 -> fc1 -> fc2
            z = x

            # 卷积层处理
            if hasattr(model, 'conv1'):
                z = model.conv1(z)
                if hasattr(model, 'bn1') and (model.use_bn if hasattr(model, 'use_bn') else False):
                    z = model.bn1(z)
                z = model.relu(z)
                z = model.maxpool(z)

            if hasattr(model, 'conv2'):
                z = model.conv2(z)
                if hasattr(model, 'bn2') and (model.use_bn if hasattr(model, 'use_bn') else False):
                    z = model.bn2(z)
                z = model.relu(z)
                z = model.maxpool(z)

            # 展平
            z = z.view(z.size(0), -1)

            # 应用dropout（如果存在）
            if hasattr(model, 'dropout'):
                z = F.dropout(z, p=model.dropout, training=model.training)

            # 第一个FC层 (这是我们需要的嵌入z)
            embedding = model.relu(model.fc1(z))

            # 再次应用dropout（如果存在）
            if hasattr(model, 'dropout'):
                z = F.dropout(embedding, p=model.dropout, training=model.training)
            else:
                z = embedding

            # 最终分类器
            pred = model.fc2(z)

            return pred, embedding
        else:
            # 后备方案: 使用完整的前向传播
            # 这不会给出正确的嵌入，但至少不会崩溃
            pred = model(x)
            logger.warning("模型没有预期的嵌入提取结构。"
                         "使用输出作为嵌入（不推荐）。")
            return pred, pred

    def _hook_on_batch_forward(self, ctx):
        """
        重写默认的前向传播钩子，实现FedLSA的核心训练逻辑

        算法流程 (对应论文Algorithm 1 Client Side):
        1. 超球面投影: h_i = norm(φ(ψ(x_i)))
        2. 计算分类损失: L_CE (公式9)
        3. 计算紧凑性损失: L_COM (公式8)
        4. 总损失: L_HC = L_CE + λ·L_COM (公式10)

        训练目标:
        - L_CE: 确保基本分类能力
        - L_COM: 在超球面上实现类内紧凑性
        """
        x, label = [_.to(ctx.device) for _ in ctx.data_batch]

        # ========== 步骤1: 前向传播获取嵌入和预测 ==========
        if self.cfg.fedlsa.use_projector:
            # 获取最终分类器之前的嵌入
            if hasattr(ctx.model, 'get_embedding'):
                # 模型有显式的嵌入提取方法 (如FedLSACNN)
                z = ctx.model.get_embedding(x)
                pred = ctx.model(x)
            else:
                # 对于标准CNN模型 (如ConvNet2)，手动提取特征
                pred, z = self._forward_with_embedding(ctx.model, x)

            # ========== 步骤2: 投影到超球面空间并归一化 ==========
            # h_i = norm(φ(z_i))，其中 ||h_i||_2 = 1
            h = self.projector(z)
            h = self.normalize_embedding(h)
        else:
            # 不使用投影器的标准前向传播
            pred = ctx.model(x)
            # 不使用投影器时无法正确计算紧凑性损失
            h = None

        if len(label.size()) == 0:
            label = label.unsqueeze(0)

        # 存储上下文变量
        ctx.y_true = CtxVar(label, LIFECYCLE.BATCH)
        ctx.y_prob = CtxVar(pred, LIFECYCLE.BATCH)
        if h is not None:
            ctx.h_embeddings = CtxVar(h, LIFECYCLE.BATCH)

        # ========== 步骤3: 计算分类损失 L_CE (公式9) ==========
        # L_CE = -1_{y_i} log(softmax(q_i))
        loss_ce = ctx.criterion(pred, label)

        # ========== 步骤4: 训练模式下计算紧凑性损失 ==========
        if ctx.cur_mode == MODE.TRAIN and h is not None:
            # 计算紧凑性损失 L_COM (公式8)
            loss_com = self.compute_compactness_loss(h, label)

            # ========== 步骤5: 总损失 L_HC (公式10) ==========
            # L_HC = L_CE + λ·L_COM
            ctx.loss_batch = CtxVar(
                loss_ce + self.lambda_com * loss_com,
                LIFECYCLE.BATCH
            )
            ctx.loss_ce = CtxVar(loss_ce, LIFECYCLE.BATCH)
            ctx.loss_com = CtxVar(loss_com, LIFECYCLE.BATCH)
        else:
            # 评估时或不使用投影器时，仅使用分类损失
            ctx.loss_batch = CtxVar(loss_ce, LIFECYCLE.BATCH)

        ctx.batch_size = CtxVar(len(label), LIFECYCLE.BATCH)

    def get_model_para(self):
        """
        获取模型参数，如果存在投影器则包含投影器参数

        返回:
            dict: 包含主模型和投影器(如果共享)的参数字典
        """
        model_para = super().get_model_para()

        if hasattr(self, 'projector') and self.cfg.fedlsa.share_projector:
            # 包含投影器参数 (用于联邦平均)
            projector_para = self._param_filter(self.projector.state_dict())
            model_para.update({f'projector.{k}': v for k, v in projector_para.items()})

        return model_para

    def update(self, model_parameters, strict=False):
        """
        更新模型参数，同时处理主模型和投影器的更新

        Args:
            model_parameters: 包含模型和投影器参数的字典
            strict: 是否严格匹配参数键
        """
        # 分离投影器参数（如果存在）
        model_params = {}
        projector_params = {}

        for key, value in model_parameters.items():
            if key.startswith('projector.'):
                projector_params[key.replace('projector.', '')] = value
            else:
                model_params[key] = value

        # 更新主模型
        if model_params:
            super().update(model_params, strict=strict)

        # 更新投影器（如果存在）
        if projector_params and hasattr(self, 'projector'):
            from federatedscope.core.auxiliaries.utils import param2tensor, merge_param_dict
            for key in projector_params:
                projector_params[key] = param2tensor(projector_params[key])
            merged_param = merge_param_dict(
                self.projector.state_dict().copy(),
                self._param_filter(projector_params)
            )
            self.projector.load_state_dict(merged_param, strict=strict)

    def _hook_on_fit_end(self, ctx):
        """
        训练结束钩子，记录FedLSA特定的指标
        """
        super()._hook_on_fit_end(ctx)

        # 如果在训练模式，记录FedLSA特定的损失
        if ctx.cur_mode == MODE.TRAIN and hasattr(ctx, 'loss_ce'):
            logger.info(f"FedLSA损失 - 分类损失(CE): {ctx.loss_ce.item():.4f}, "
                       f"紧凑性损失(COM): {ctx.loss_com.item():.4f}")


def call_fedlsa_trainer(trainer_type):
    """
    FedLSA Trainer注册的辅助函数

    Args:
        trainer_type: trainer类型字符串

    Returns:
        FedLSATrainer类或None
    """
    if trainer_type.lower() == 'fedlsa_trainer':
        return FedLSATrainer
    else:
        return None


# 注册FedLSA trainer到FederatedScope框架
from federatedscope.register import register_trainer
register_trainer('fedlsa_trainer', call_fedlsa_trainer)




