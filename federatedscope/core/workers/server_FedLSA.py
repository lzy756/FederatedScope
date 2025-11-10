"""
FedLSA (联邦可学习语义锚点) Server实现

参考论文:
Beyond Federated Prototype Learning: Learnable Semantic Anchors with
Hyperspherical Contrast for Domain-Skewed Data (AAAI 2025)

核心功能:
1. 语义锚点学习: 通过网络训练直接学习类别表示
2. 锚点交叉熵损失: 确保每个锚点对应唯一语义
3. 分离损失: 增大锚点间的边界，提升类间可分性
4. 锚点分发: 将学习好的锚点广播给客户端
"""
import logging
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from federatedscope.core.workers import Server
from federatedscope.core.message import Message

logger = logging.getLogger(__name__)


class SemanticAnchorLearner(nn.Module):
    """
    服务器端语义锚点学习模块

    锚点生成流程:
    1. 随机初始化锚点参数 R: [C, D]
    2. 通过映射函数 Θ (2层MLP) 生成锚点: A = Θ(R)
    3. L2归一化到单位超球面: A_norm = A / ||A||_2

    优势:
    - 锚点通过网络学习，独立于具体域
    - 避免传统原型学习中的域特定信息污染
    - 可通过梯度优化获得最优类别表示
    """

    def __init__(self, num_classes, embedding_dim, device):
        super(SemanticAnchorLearner, self).__init__()
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim

        # 初始化随机锚点参数 R
        # 形状: [C, D]，其中 C 是类别数，D 是嵌入维度
        self.anchor_params = nn.Parameter(
            torch.randn(num_classes, embedding_dim, device=device)
        )

        # 映射函数 Θ (2层MLP)
        # 将参数 R 映射到语义锚点 A
        self.mapping_func = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim)
        ).to(device)

    def forward(self):
        """
        从参数生成语义锚点

        生成步骤:
        1. A = Θ(R): 通过映射函数生成初始锚点
        2. A_norm = A / ||A||_2: 归一化到单位超球面

        Returns:
            anchors: 语义锚点张量，形状 [C, D]
                    C: 类别数量
                    D: 超球面嵌入维度
                    满足 ||anchors[i]||_2 = 1
        """
        # 步骤1: A = Θ(R)
        anchors = self.mapping_func(self.anchor_params)

        # 步骤2: 归一化到单位超球面
        anchors = F.normalize(anchors, p=2, dim=1)

        return anchors


class FedLSAServer(Server):
    """
    FedLSA方法的服务器端实现

    核心职责:
    1. 语义锚点学习: 优化锚点参数 R 和映射函数 Θ
    2. 锚点损失计算:
       - L_ACE: 锚点交叉熵损失 (公式3)
       - L_SEP: 分离损失 (公式4)
       - L_LSA = L_ACE + α·L_SEP (公式5)
    3. 客户端模型聚合: FedAvg
    4. 参数分发: 广播锚点和全局模型给客户端

    算法流程 (对应论文Algorithm 1 Server Side):
    - 每个通信轮次开始前训练锚点 (E_s轮)
    - 聚合客户端上传的模型
    - 广播更新后的锚点和全局模型
    """

    def __init__(self,
                 ID=-1,
                 state=0,
                 config=None,
                 data=None,
                 model=None,
                 client_num=5,
                 total_round_num=10,
                 device='cpu',
                 strategy=None,
                 unseen_clients_id=None,
                 **kwargs):
        super(FedLSAServer, self).__init__(
            ID=ID,
            state=state,
            config=config,
            data=data,
            model=model,
            client_num=client_num,
            total_round_num=total_round_num,
            device=device,
            strategy=strategy,
            unseen_clients_id=unseen_clients_id,
            **kwargs
        )

        # FedLSA特定参数
        self.alpha_sep = config.fedlsa.alpha_sep  # α: 分离损失权重 (公式5)
        self.tau = config.fedlsa.tau  # τ: 温度参数 (公式4)
        self.num_classes = config.model.num_classes  # C: 类别数量
        self.anchor_train_epochs = config.fedlsa.anchor_train_epochs  # E_s: 锚点训练轮数
        self.anchor_lr = config.fedlsa.anchor_lr  # η: 锚点学习率

        # 获取嵌入维度 (超球面空间的维度)
        if hasattr(config.fedlsa, 'projector_output_dim'):
            embedding_dim = config.fedlsa.projector_output_dim  # L: 超球面嵌入维度
        else:
            embedding_dim = config.model.hidden  # 回退到模型隐藏维度

        # 初始化语义锚点学习器
        # 包含: 锚点参数 R 和映射函数 Θ
        self.anchor_learner = SemanticAnchorLearner(
            num_classes=self.num_classes,
            embedding_dim=embedding_dim,
            device=device
        )

        # 创建语义感知分类器 φ_glo
        # 用于将超球面锚点映射到类别logits，实现语义区分
        self.semantic_classifier = nn.Linear(embedding_dim, self.num_classes).to(device)

        # 锚点学习优化器 (同时优化 R, Θ, φ_glo)
        self.anchor_optimizer = torch.optim.Adam(
            list(self.anchor_learner.parameters()) + list(self.semantic_classifier.parameters()),
            lr=self.anchor_lr
        )

        # 存储当前训练好的锚点
        self.semantic_anchors = None

        logger.info(f"FedLSAServer初始化完成: alpha_sep={self.alpha_sep}, "
                   f"tau={self.tau}, anchor_train_epochs={self.anchor_train_epochs}")

    def compute_anchor_ce_loss(self, anchors):
        """
        计算锚点交叉熵损失 (L_ACE)

        论文公式3:
        L_ACE = -1_{y_i} log(softmax(ρ_i))
        其中 ρ_i = φ_glo(a_i)

        物理意义:
        - 使用全局语义感知分类器 φ_glo 区分不同类别的锚点
        - 确保每个锚点对应唯一的语义类别
        - φ_glo 融合了多域信息，提供泛化的语义判别能力

        实现细节:
        1. 将锚点通过语义分类器映射到logits
        2. 为每个锚点创建one-hot标签 (第i个锚点对应第i类)
        3. 计算交叉熵损失，优化锚点的语义区分度

        Args:
            anchors: 语义锚点 [C, D]，在超球面空间
                    C: 类别数量
                    D: 超球面嵌入维度

        Returns:
            损失标量
        """
        # 使用语义分类器将锚点映射到logits
        # anchors: [C, D] -> logits: [C, C]
        logits = self.semantic_classifier(anchors)

        # 创建one-hot标签: 第0个锚点→类别0, 第1个锚点→类别1, ...
        labels = torch.arange(self.num_classes, device=anchors.device)

        # 计算交叉熵损失
        # 这会强制每个锚点只激活对应的类别输出
        loss = F.cross_entropy(logits, labels)

        return loss

    def compute_separation_loss(self, anchors):
        """
        计算分离损失 (L_SEP)，推开不同的锚点

        论文公式4:
        L_SEP = log( Σ_{j≠i} exp(a_i·a_j^T / τ) / (C-1) )

        物理意义:
        - 鼓励不同的语义锚点在超球面上相互分离
        - 增大锚点间的边界，提升类间可分性
        - 对比学习风格: 最小化锚点间的相似度

        实现细节:
        1. 计算锚点间的成对相似度 (内积，因为已归一化)
        2. 温度缩放: sim / τ
        3. 对每个锚点 i，计算与其他锚点 j≠i 的相似度之和
        4. 使用log-sum-exp技巧确保数值稳定性

        为什么要推开锚点?
        - 传统方法: 全局原型通过平均生成，边界会衰减
        - FedLSA方法: 显式优化分离度，确保足够的类间边界
        - 结果: 客户端学习到的表示具有更好的类间可分性

        Args:
            anchors: 语义锚点 [C, D]
                    C: 类别数量
                    D: 超球面嵌入维度

        Returns:
            损失标量
        """
        # 计算锚点间的成对相似度
        # anchors: [C, D], similarity_matrix: [C, C]
        # 注意: 锚点已经归一化，所以内积 = 余弦相似度
        similarity_matrix = torch.matmul(anchors, anchors.T)  # [C, C]

        # 温度缩放 (控制分布的尖锐程度)
        similarity_matrix = similarity_matrix / self.tau

        # 对于每个锚点，我们要最小化与其他锚点的相似度
        # 使用对比学习风格的损失:
        # L_i = log(Σ_{j≠i} exp(sim(a_i, a_j) / τ))
        # 这会鼓励锚点在超球面上远离彼此
        C = anchors.size(0)

        # 创建掩码排除自相似度 (对角线元素)
        # mask[i,i] = 0, mask[i,j] = 1 for i != j
        mask = 1.0 - torch.eye(C, device=anchors.device)

        # 应用掩码: 将对角线设为极小值，使 exp() ≈ 0
        # 这样自相似度就不会包含在求和中
        masked_similarity = similarity_matrix * mask + (1.0 - mask) * (-1e9)

        # 对每个锚点计算 log-sum-exp (数值稳定版本)
        # logsumexp 比 log(sum(exp())) 更稳定
        loss_per_anchor = torch.logsumexp(masked_similarity, dim=1)

        # 对所有锚点取平均
        loss = torch.mean(loss_per_anchor)

        return loss

    def train_semantic_anchors(self):
        """
        服务器端训练语义锚点

        论文公式5:
        L_LSA = L_ACE + α·L_SEP

        算法流程 (对应论文Algorithm 1 Server Side):
        for e = 1 to E_s:
            1. A = Θ(R): 生成当前锚点
            2. L_ACE = -1_{y_i} log(softmax(φ_glo(a_i))): 计算锚点CE损失 (公式3)
            3. L_SEP = log(Σ_{j≠i} exp(a_i·a_j^T/τ) / (C-1)): 计算分离损失 (公式4)
            4. L_LSA = L_ACE + α·L_SEP: 总损失 (公式5)
            5. 反向传播更新 R, Θ, φ_glo

        训练目标:
        - L_ACE: 确保锚点的语义区分度
        - L_SEP: 增大锚点间的边界
        - 结果: 获得域无关、边界清晰的类别表示
        """
        self.anchor_learner.train()

        # 记录初始损失和最终损失，用于诊断
        initial_losses = None

        for epoch in range(self.anchor_train_epochs):
            self.anchor_optimizer.zero_grad()

            # 步骤1: 生成当前锚点 A = Θ(R)
            anchors = self.anchor_learner()

            # 步骤2: 计算锚点CE损失 L_ACE (公式3)
            loss_ace = self.compute_anchor_ce_loss(anchors)

            # 步骤3: 计算分离损失 L_SEP (公式4)
            loss_sep = self.compute_separation_loss(anchors)

            # 步骤4: 总损失 L_LSA (公式5)
            loss_total = loss_ace + self.alpha_sep * loss_sep

            # 记录初始损失
            if initial_losses is None:
                initial_losses = {
                    'ace': loss_ace.item(),
                    'sep': loss_sep.item(),
                    'total': loss_total.item()
                }

            # 步骤5: 反向传播更新参数
            # R^e ← R^{e-1} - η∇L_LSA
            # Θ^e ← Θ^{e-1} - η∇L_LSA
            loss_total.backward()
            self.anchor_optimizer.step()

            # 每5轮或最后一轮打印日志
            if epoch % 1 == 0 or epoch == self.anchor_train_epochs - 1:
                # 计算锚点间的平均距离（诊断信息）
                with torch.no_grad():
                    # ⚠️ 重要：detach锚点，避免梯度累积导致显存泄漏
                    anchors_detached = anchors.detach()
                    sim_matrix = torch.matmul(anchors_detached, anchors_detached.T)
                    # 排除对角线
                    mask = 1.0 - torch.eye(self.num_classes, device=anchors.device)
                    avg_similarity = (sim_matrix * mask).sum() / (self.num_classes * (self.num_classes - 1))
                    min_similarity = (sim_matrix + torch.eye(self.num_classes, device=anchors.device) * 2).min()

                logger.info(f"锚点训练 epoch {epoch}/{self.anchor_train_epochs-1}: "
                           f"L_ACE={loss_ace.item():.4f}, "
                           f"L_SEP={loss_sep.item():.4f}, "
                           f"L_LSA={loss_total.item():.4f} | "
                           f"锚点平均相似度={avg_similarity.item():.4f}, "
                           f"最小相似度={min_similarity.item():.4f}")

        # 存储训练好的锚点 (用于分发给客户端)
        self.anchor_learner.eval()
        with torch.no_grad():
            self.semantic_anchors = self.anchor_learner()

        # 打印训练摘要
        final_losses = {
            'ace': loss_ace.item(),
            'sep': loss_sep.item(),
            'total': loss_total.item()
        }
        logger.info(f"语义锚点训练完成 | "
                   f"L_ACE: {initial_losses['ace']:.4f}→{final_losses['ace']:.4f} "
                   f"(下降{initial_losses['ace']-final_losses['ace']:.4f}), "
                   f"L_SEP: {initial_losses['sep']:.4f}→{final_losses['sep']:.4f} "
                   f"(下降{initial_losses['sep']-final_losses['sep']:.4f})")

    def broadcast_model_para(self,
                             msg_type='model_para',
                             sample_client_num=-1,
                             filter_unseen_clients=True):
        """
        重写以广播模型参数和语义锚点

        流程:
        1. 确定接收客户端列表
        2. 广播全局模型参数 (FedAvg聚合结果)
        3. 额外广播语义锚点 (仅在model_para类型时)

        这确保客户端同时收到:
        - 更新的全局模型 Φ_glo
        - 服务器学习的语义锚点 A
        """
        # 首先确定接收者列表 (与父类相同的逻辑)
        if filter_unseen_clients:
            # 采样时过滤掉未见过的客户端
            self.sampler.change_state(self.unseen_clients_id, 'unseen')

        if sample_client_num > 0:
            receiver = self.sampler.sample(size=sample_client_num)
        else:
            # 广播给所有客户端
            receiver = list(self.comm_manager.neighbors.keys())
            if msg_type == 'model_para':
                self.sampler.change_state(receiver, 'working')

        # 现在使用父类的方法广播模型参数
        # 但通过直接调用广播逻辑来跳过采样部分
        if self._noise_injector is not None and msg_type == 'model_para':
            # 仅在广播参数时注入噪声
            for model_idx_i in range(len(self.models)):
                num_sample_clients = [
                    v["num_sample"] for v in self.join_in_info.values()
                ]
                self._noise_injector(self._cfg, num_sample_clients,
                                     self.models[model_idx_i])

        skip_broadcast = self._cfg.federate.method in ["local", "global"]
        if self.model_num > 1:
            model_para = [{} if skip_broadcast else model.state_dict()
                          for model in self.models]
        else:
            model_para = {} if skip_broadcast else self.models[0].state_dict()

        # 量化
        if msg_type == 'model_para' and not skip_broadcast and \
                self._cfg.quantization.method == 'uniform':
            from federatedscope.core.compression import \
                symmetric_uniform_quantization
            nbits = self._cfg.quantization.nbits
            if self.model_num > 1:
                model_para = [
                    symmetric_uniform_quantization(x, nbits)
                    for x in model_para
                ]
            else:
                model_para = symmetric_uniform_quantization(model_para, nbits)

        # 我们定义评估发生在epoch结束时
        rnd = self.state - 1 if msg_type == 'evaluate' else self.state

        self.comm_manager.send(
            Message(msg_type=msg_type,
                    sender=self.ID,
                    receiver=receiver,
                    state=min(rnd, self.total_round_num),
                    timestamp=self.cur_timestamp,
                    content=model_para))
        if self._cfg.federate.online_aggr:
            for idx in range(self.model_num):
                self.aggregators[idx].reset()

        # 额外广播语义锚点 (如果存在，且消息类型为model_para)
        # 注意: 锚点在调用此方法之前的 _start_new_training_round 中训练
        if self.semantic_anchors is not None and msg_type == 'model_para':
            self.broadcast_semantic_anchors(receiver=receiver)

        if filter_unseen_clients:
            # 恢复采样器中未见客户端的状态
            self.sampler.change_state(self.unseen_clients_id, 'seen')

    def broadcast_semantic_anchors(self, receiver=None):
        """
        向客户端广播语义锚点

        锚点分发策略:
        - 锚点在服务器端训练，独立于客户端域
        - 通过单独的消息发送 (msg_type='semantic_anchors')
        - 客户端接收后用于指导超球面对比学习

        Args:
            receiver: 接收客户端ID列表。如果为None，广播给所有邻居。
        """
        if receiver is None:
            # 广播给所有邻居
            receiver = list(self.comm_manager.neighbors.keys())

        if not receiver:
            logger.warning("没有可用的接收者用于语义锚点广播")
            return

        # 作为单独的消息发送锚点
        self.comm_manager.send(
            Message(msg_type='semantic_anchors',
                    sender=self.ID,
                    receiver=receiver,
                    state=self.state,
                    timestamp=self.cur_timestamp,
                    content=self.semantic_anchors.detach().cpu())
        )

        logger.debug(f"已向{len(receiver)}个客户端广播语义锚点")

    def _start_new_training_round(self, aggregated_num=0):
        """
        重写以在开始新轮次前训练锚点

        FedLSA的通信轮次流程:
        1. 服务器训练语义锚点 (E_s轮优化)
        2. 广播锚点和全局模型给客户端
        3. 客户端本地训练 (使用锚点指导)
        4. 客户端上传模型参数
        5. 服务器聚合模型 (FedAvg)
        6. 进入下一轮
        """
        # 在分发给客户端之前训练锚点
        # 对于第一轮 (state=0)，初始化锚点
        # 对于后续轮次，更新锚点
        self.train_semantic_anchors()

        # 调用父类的方法 (会触发broadcast_model_para)
        super()._start_new_training_round(aggregated_num)

    def trigger_for_feat_engr(self,
                              trigger_train_func,
                              kwargs_for_trigger_train_func={}):
        """
        重写以在第一轮之前训练锚点

        特征工程触发器:
        - 在联邦学习开始前调用
        - 用于初始化语义锚点
        """
        # 在开始第一轮前训练锚点
        self.train_semantic_anchors()

        # 调用原始触发函数 (会广播模型和锚点)
        super().trigger_for_feat_engr(trigger_train_func, kwargs_for_trigger_train_func)






