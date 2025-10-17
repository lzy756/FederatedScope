# FedLSA 代码实现完整技术文档

## 目录
1. [论文概述](#论文概述)
2. [核心理论与公式](#核心理论与公式)
3. [架构设计](#架构设计)
4. [代码实现详解](#代码实现详解)
5. [配置系统](#配置系统)
6. [完整训练流程](#完整训练流程)

---

## 论文概述

**标题**: Beyond Federated Prototype Learning: Learnable Semantic Anchors with Hyperspherical Contrast for Domain-Skewed Data

**发表**: AAAI 2025

**核心问题**:
传统联邦原型学习在面对域偏斜数据时存在两个主要缺陷:
1. **边界衰减问题**: 本地原型通过类内嵌入平均获得,携带域特定标记,导致聚合后的全局原型间边界减弱
2. **分布不均问题**: 本地域偏斜嵌入在欧几里得空间可能不呈现均匀分布,不利于基于原型的对比学习

**解决方案**: FedLSA (Federated Learnable Semantic Anchors)
- 直接学习语义锚点而非通过平均生成原型
- 在超球面空间进行表示学习,确保均匀分布
- 通过对比损失实现类内紧凑性和类间可分性

---

## 核心理论与公式

### 2.1 问题定义

联邦学习系统包含中心服务器和 M 个客户端。每个客户端 m 有本地模型:

$$\Phi_m = \psi_m \circ \varphi_m \circ \phi_m$$

其中:
- $\psi_m$: 编码器,将原始数据映射到嵌入空间 $z_i = \psi_m(x_i) \in \mathbb{R}^I$
- $\varphi_m$: 投影器,映射到低维超球面空间 $h_i = \text{nor}(\varphi_m(z_i)) \in \mathbb{R}^L$
- $\phi_m$: 分类器,映射到类别输出 $q_i = \phi_m(h_i) \in \mathbb{R}^C$
- $\text{nor}(x) = x / \|x\|_2$: L2归一化

**域偏斜定义**: $P_m(x|y) \neq P_n(x|y)$ (不同客户端的条件分布不同)

### 2.2 传统联邦原型学习 (对比)

传统方法中,第 m 个客户端的第 c 类原型定义为:

$$p_m^c = \frac{1}{|D_m^c|} \sum_{i=1}^{|D_m^c|} \psi_m(x_i) \quad \text{(公式1)}$$

$$P_m = [p_m^1; \ldots; p_m^C] \in \mathbb{R}^{C \times I}$$

全局原型通过服务器端聚合得到:

$$p^c = \frac{1}{M} \sum_{m=1}^{M} p_m^c \quad \text{(公式2)}$$

$$\mathcal{P} = [p^1; \ldots; p^C] \in \mathbb{R}^{C \times I}$$

**问题**: 这种平均方式会保留域特定信息,导致全局原型间边界减弱。

---

### 2.3 语义锚点学习 (Semantic Anchors Learning)

#### 2.3.1 锚点初始化与映射

FedLSA 直接学习语义锚点集合:

$$A = [a_1; \ldots; a_C]$$

锚点由随机参数 $R \in \mathbb{R}^{C \times I}$ 通过映射函数 $\Theta(\cdot)$ 生成:

$$A = \Theta(R)$$

其中 $\Theta$ 是一个双层MLP (Multi-Layer Perceptron)。

**代码实现位置**: `server_FedLSA.py:SemanticAnchorLearner`

```python
class SemanticAnchorLearner(nn.Module):
    def __init__(self, num_classes, embedding_dim, device):
        super().__init__()
        # 初始化随机参数 R
        self.anchor_params = nn.Parameter(
            torch.randn(num_classes, embedding_dim, device=device)
        )

        # 映射函数 Theta (2层MLP)
        self.mapping_func = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim)
        )

    def forward(self):
        # A = Theta(R)
        anchors = self.mapping_func(self.anchor_params)
        # 归一化到单位超球面
        anchors = F.normalize(anchors, p=2, dim=1)
        return anchors
```

#### 2.3.2 锚点交叉熵损失 (Anchor CrossEntropy Loss)

使用全局分类器 $\phi_{glo}$ 区分不同类别的锚点:

$$\mathcal{L}_{ACE} = -\mathbf{1}_{y_i} \log(\text{softmax}(\rho_i)) \quad \text{(公式3)}$$

其中 $\rho_i = \phi_{glo}(a_i)$ 是第 i 个锚点的预测logits。

**目的**: 确保每个锚点对应唯一的语义类别。

**代码实现**: `server_FedLSA.py:compute_anchor_ce_loss`

```python
def compute_anchor_ce_loss(self, anchors):
    """
    公式3实现: L_ACE = -1_{y_i} log(softmax(ρ_i))
    """
    # 使用语义感知分类器映射锚点到logits
    # anchors: [C, D] -> logits: [C, C]
    logits = self.semantic_classifier(anchors)

    # 创建one-hot标签 [0, 1, 2, ..., C-1]
    labels = torch.arange(self.num_classes, device=anchors.device)

    # 计算交叉熵损失
    loss = F.cross_entropy(logits, labels)
    return loss
```

#### 2.3.3 分离损失 (Separation Loss)

为了增强锚点间的边界,引入分离损失:

$$\mathcal{L}_{SEP} = \log \frac{\sum_{j=1,j\neq i}^{C} \exp(a_i a_j^T / \tau)}{C-1} \quad \text{(公式4)}$$

其中 $\tau$ 是温度参数。

**目的**: 推开不同的锚点,增大类间分离度。

**代码实现**: `server_FedLSA.py:compute_separation_loss`

```python
def compute_separation_loss(self, anchors):
    """
    公式4实现: L_SEP = log(sum_{j≠i} exp(a_i·a_j^T / τ) / (C-1))
    """
    # 计算锚点间的成对相似度
    # anchors: [C, D], similarity_matrix: [C, C]
    similarity_matrix = torch.matmul(anchors, anchors.T)  # 已归一化

    # 温度缩放
    similarity_matrix = similarity_matrix / self.tau

    C = anchors.size(0)

    # 创建掩码排除自相似度(对角元素)
    mask = 1.0 - torch.eye(C, device=anchors.device)

    # 将对角线设为极小值,使exp()≈0
    masked_similarity = similarity_matrix * mask + (1.0 - mask) * (-1e9)

    # 计算每个锚点的log-sum-exp (数值稳定版本)
    loss_per_anchor = torch.logsumexp(masked_similarity, dim=1)

    # 对所有锚点取平均
    loss = torch.mean(loss_per_anchor)
    return loss
```

#### 2.3.4 总锚点学习损失

$$\mathcal{L}_{LSA} = \mathcal{L}_{ACE} + \alpha \mathcal{L}_{SEP} \quad \text{(公式5)}$$

其中 $\alpha$ 是权衡参数。

**代码实现**: `server_FedLSA.py:train_semantic_anchors`

```python
def train_semantic_anchors(self):
    """
    服务器端训练语义锚点
    优化目标: L_LSA = L_ACE + α·L_SEP (公式5)
    """
    self.anchor_learner.train()

    for epoch in range(self.anchor_train_epochs):
        self.anchor_optimizer.zero_grad()

        # 生成当前锚点
        anchors = self.anchor_learner()

        # 计算锚点交叉熵损失 (公式3)
        loss_ace = self.compute_anchor_ce_loss(anchors)

        # 计算分离损失 (公式4)
        loss_sep = self.compute_separation_loss(anchors)

        # 总损失 (公式5)
        loss_total = loss_ace + self.alpha_sep * loss_sep

        # 反向传播
        loss_total.backward()
        self.anchor_optimizer.step()
```

---

### 2.4 超球面表示学习 (Hyperspherical Representation Learning)

#### 2.4.1 von Mises-Fisher 分布

超球面空间可建模为 von Mises-Fisher 分布:

$$p_D(h; a_c, \kappa) = N_D(\kappa) \exp(\kappa a_c^T h) \quad \text{(公式6)}$$

其中:
- $h$: 超球面嵌入
- $a_c$: 均值(即第 c 类的锚点)
- $\kappa$: 集中度参数(控制围绕均值的紧密程度)
- $N_D(\kappa)$: 归一化因子

#### 2.4.2 类别分配概率

第 i 个超球面嵌入 $h_i$ 被分配到第 c 类的归一化概率:

$$p(y_i=c | h_i; \{\kappa, a_j\}_{j=1}^C) = \frac{\exp(a_c^T h_i / \tau)}{\sum_{j=1}^{C} \exp(a_j^T h_i / \tau)} \quad \text{(公式7)}$$

其中 $\tau = 1/\kappa$ 是温度参数。

#### 2.4.3 紧凑性损失 (Compactness Loss)

通过最大似然估计(MLE),采用负对数似然策略:

$$\mathcal{L}_{COM} = -\log \frac{\exp(a_{y_i}^T h_i / \tau)}{\sum_{j=1}^{C} \exp(a_j^T h_i / \tau)} \quad \text{(公式8)}$$

其中 $a_{y_i}$ 是 $h_i$ 对应的锚点。

**目的**: 将每个超球面嵌入拉向其对应的锚点,实现类内紧凑性。

**代码实现**: `trainer_FedLSA.py:compute_compactness_loss`

```python
def compute_compactness_loss(self, h_embeddings, labels):
    """
    公式8实现: L_COM = -log(exp(a_{y_i}^T h_i / τ) / Σ_j exp(a_j^T h_i / τ))
    基于von Mises-Fisher分布的对比损失

    Args:
        h_embeddings: 归一化的超球面嵌入 [B, D]
        labels: 真实标签 [B]

    Returns:
        紧凑性损失标量
    """
    if self.semantic_anchors is None:
        return torch.tensor(0.0, device=h_embeddings.device)

    # 计算嵌入和所有锚点的相似度
    # h_embeddings: [B, D], semantic_anchors: [C, D]
    # logits: [B, C]
    logits = torch.matmul(h_embeddings, self.semantic_anchors.T) / self.tau

    # 计算负对数似然 (等价于交叉熵)
    loss = F.cross_entropy(logits, labels)

    return loss
```

#### 2.4.4 分类损失 (Classification Loss)

标准交叉熵损失,赋予模型基本分类能力:

$$\mathcal{L}_{CE} = -\mathbf{1}_{y_i} \log(\text{softmax}(q_i)) \quad \text{(公式9)}$$

其中 $q_i = \phi_m(h_i)$ 是预测的logits输出。

#### 2.4.5 总客户端损失

$$\mathcal{L}_{HC} = \mathcal{L}_{CE} + \lambda \mathcal{L}_{COM} \quad \text{(公式10)}$$

其中 $\lambda$ 是权衡参数。

**代码实现**: `trainer_FedLSA.py:_hook_on_batch_forward`

```python
def _hook_on_batch_forward(self, ctx):
    """
    前向传播钩子,计算总损失 L_HC = L_CE + λ·L_COM (公式10)
    """
    x, label = [_.to(ctx.device) for _ in ctx.data_batch]

    # 前向传播获取嵌入和预测
    pred, z = self._forward_with_embedding(ctx.model, x)

    # 投影到超球面空间并归一化
    h = self.projector(z)
    h = self.normalize_embedding(h)  # h = z / ||z||_2

    # 存储上下文变量
    ctx.y_true = CtxVar(label, LIFECYCLE.BATCH)
    ctx.y_prob = CtxVar(pred, LIFECYCLE.BATCH)
    ctx.h_embeddings = CtxVar(h, LIFECYCLE.BATCH)

    # 计算分类损失 L_CE (公式9)
    loss_ce = ctx.criterion(pred, label)

    # 训练模式下计算紧凑性损失 L_COM (公式8)
    if ctx.cur_mode == MODE.TRAIN:
        loss_com = self.compute_compactness_loss(h, label)

        # 总损失 L_HC (公式10)
        ctx.loss_batch = CtxVar(
            loss_ce + self.lambda_com * loss_com,
            LIFECYCLE.BATCH
        )
        ctx.loss_ce = CtxVar(loss_ce, LIFECYCLE.BATCH)
        ctx.loss_com = CtxVar(loss_com, LIFECYCLE.BATCH)
    else:
        # 评估时仅使用分类损失
        ctx.loss_batch = CtxVar(loss_ce, LIFECYCLE.BATCH)
```

---

## 架构设计

### 3.1 整体架构

```
┌─────────────────────────────────────────────────────────────┐
│                         FedLSA System                        │
├──────────────────────────┬──────────────────────────────────┤
│      Server Side         │         Client Side (×M)         │
│                          │                                  │
│  ┌────────────────────┐  │  ┌────────────────────────────┐ │
│  │ SemanticAnchor     │  │  │  Local Model (Φ_m)         │ │
│  │ Learner            │  │  │                            │ │
│  │  - anchor_params R │  │  │  ┌──────────────────────┐ │ │
│  │  - mapping_func Θ  │  │  │  │ Encoder ψ_m          │ │ │
│  │                    │  │  │  │  (Conv1, Conv2)      │ │ │
│  │  A = Θ(R)          │  │  │  │  z = ψ_m(x)          │ │ │
│  │  A_norm = A/||A||  │  │  │  └──────────────────────┘ │ │
│  └────────────────────┘  │  │             ↓              │ │
│            │             │  │  ┌──────────────────────┐ │ │
│            ↓             │  │  │ Projector φ_m        │ │ │
│  ┌────────────────────┐  │  │  │  (2-layer MLP)       │ │ │
│  │ Semantic-Aware     │  │  │  │  h = φ_m(z)          │ │ │
│  │ Classifier φ_glo   │  │  │  │  h_norm = h/||h||    │ │ │
│  │                    │  │  │  └──────────────────────┘ │ │
│  │  ρ = φ_glo(A)      │  │  │             ↓              │ │
│  └────────────────────┘  │  │  ┌──────────────────────┐ │ │
│            │             │  │  │ Classifier φ_m       │ │ │
│  Loss Computation:       │  │  │  q = φ_m(h)          │ │ │
│  L_ACE = CE(ρ, labels)   │  │  └──────────────────────┘ │ │
│  L_SEP = log(Σexp(a·aᵀ)) │  │                            │ │
│  L_LSA = L_ACE + α·L_SEP │  │  Loss Computation:         │ │
│                          │  │  L_CE = CE(q, y)           │ │
│  ↓ Optimize R, Θ, φ_glo  │  │  L_COM = CE(h·Aᵀ/τ, y)     │ │
│  ↓ Broadcast A, Φ_glo    │  │  L_HC = L_CE + λ·L_COM     │ │
│                          │  │                            │ │
└──────────────────────────┴──────────────────────────────────┘
```

### 3.2 模型架构

**FedLSACNN** (federatedscope/cv/model/fedlsa_cnn.py:17)

```python
FedLSACNN(
    in_channels=3,      # 输入通道数
    h=32, w=32,         # 输入尺寸
    hidden=512,         # 隐藏维度(嵌入维度)
    class_num=10,       # 类别数
    use_bn=True,        # 是否使用批归一化
    dropout=0.0         # Dropout率
)
```

**层次结构**:
1. **编码器 (ψ_m)**:
   - Conv1: `3×32×32 → 32×32×32` (kernel=5, padding=2)
   - BN1 + ReLU + MaxPool2: `32×32×32 → 32×16×16`
   - Conv2: `32×16×16 → 64×16×16` (kernel=5, padding=2)
   - BN2 + ReLU + MaxPool2: `64×16×16 → 64×8×8`
   - Flatten: `64×8×8 → 4096`
   - FC1: `4096 → 512` (输出嵌入 z)

2. **投影器 (φ_m)** (在Trainer中):
   - Linear1: `512 → 512`
   - ReLU
   - Linear2: `512 → 128` (输出超球面嵌入 h)
   - Normalize: `h = h / ||h||_2`

3. **分类器 (φ_m)**:
   - FC2: `512 → 256`
   - FC3: `256 → 10` (输出类别logits q)

### 3.3 核心模块

#### 3.3.1 Server: FedLSAServer
- **文件**: `federatedscope/core/workers/server_FedLSA.py:56`
- **职责**:
  1. 训练语义锚点 (公式3-5)
  2. 聚合客户端模型
  3. 广播锚点和全局模型

**关键方法**:
- `train_semantic_anchors()`: 优化锚点参数 (算法1 Server Side)
- `compute_anchor_ce_loss()`: 计算 L_ACE
- `compute_separation_loss()`: 计算 L_SEP
- `broadcast_semantic_anchors()`: 分发锚点给客户端

#### 3.3.2 Client: FedLSAClient
- **文件**: `federatedscope/core/workers/client_FedLSA.py:11`
- **职责**:
  1. 接收锚点和全局模型
  2. 本地训练
  3. 上传模型参数

**关键方法**:
- `callback_funcs_for_semantic_anchors()`: 处理接收的锚点

#### 3.3.3 Trainer: FedLSATrainer
- **文件**: `federatedscope/core/trainers/trainer_FedLSA.py:19`
- **职责**:
  1. 超球面投影
  2. 计算紧凑性损失 (公式8)
  3. 梯度更新

**关键方法**:
- `_hook_on_batch_forward()`: 前向传播钩子,计算总损失 (公式10)
- `compute_compactness_loss()`: 计算 L_COM (公式8)
- `normalize_embedding()`: L2归一化到单位球面
- `_forward_with_embedding()`: 提取中间嵌入

---

## 代码实现详解

### 4.1 超球面投影实现

**理论**: 归一化操作 $\text{nor}(x) = x / \|x\|_2$

```python
# trainer_FedLSA.py:83
def normalize_embedding(self, embeddings):
    """
    归一化嵌入到单位超球面

    Args:
        embeddings: [B, D] 张量

    Returns:
        归一化嵌入,满足 ||h||_2 = 1
    """
    return F.normalize(embeddings, p=2, dim=1)
```

**使用场景**:
1. 投影器输出后归一化 (客户端)
2. 锚点生成后归一化 (服务器)

### 4.2 锚点学习完整流程

**服务器端算法** (算法1 Server Side):

```python
# server_FedLSA.py:192
def train_semantic_anchors(self):
    """
    算法1 Server Side: 语义锚点训练

    Input:
        - R: 锚点初始化参数 [C, D]
        - E_s: 服务器训练轮数 (anchor_train_epochs)
        - η: 学习率 (anchor_lr)
        - α: 分离损失权重 (alpha_sep)
        - τ: 温度参数 (tau)

    Output:
        - A: 训练好的语义锚点 [C, D]
    """
    self.anchor_learner.train()

    for epoch in range(self.anchor_train_epochs):  # E_s
        self.anchor_optimizer.zero_grad()

        # Step 1: A = Θ(R)
        anchors = self.anchor_learner()

        # Step 2: 计算 L_ACE (公式3)
        # L_ACE ← (softmax(φ_glo(a^i)), y_i)
        loss_ace = self.compute_anchor_ce_loss(anchors)

        # Step 3: 计算 L_SEP (公式4)
        # L_SEP ← ({a_j}_{j=1}^C)
        loss_sep = self.compute_separation_loss(anchors)

        # Step 4: 总损失 L_LSA (公式5)
        loss_total = loss_ace + self.alpha_sep * loss_sep

        # Step 5: 更新参数
        # R^e ← R^{e-1} - η∇L_LSA
        # Θ^e ← Θ^{e-1} - η∇L_LSA
        loss_total.backward()
        self.anchor_optimizer.step()

    # Step 6: 存储训练好的锚点
    self.anchor_learner.eval()
    with torch.no_grad():
        self.semantic_anchors = self.anchor_learner()
```

### 4.3 客户端训练完整流程

**客户端算法** (算法1 Client Side):

```python
# trainer_FedLSA.py:228
def _hook_on_batch_forward(self, ctx):
    """
    算法1 Client Side: 客户端训练前向传播

    Input:
        - x_i, y_i: 输入数据和标签
        - {a_j}_{j=1}^C: 从服务器接收的语义锚点
        - λ: 紧凑性损失权重 (lambda_com)
        - τ: 温度参数 (tau)

    Process:
        1. 超球面投影
        2. 计算 L_CE
        3. 计算 L_COM
        4. 总损失 L_HC
    """
    x, label = [_.to(ctx.device) for _ in ctx.data_batch]

    # Step 1: 前向传播获取嵌入 z
    pred, z = self._forward_with_embedding(ctx.model, x)

    # Step 2: 投影到超球面空间
    # h_i = nor(φ(ψ(x_i)))
    h = self.projector(z)
    h = self.normalize_embedding(h)  # 归一化到单位球面

    # Step 3: 计算分类损失 L_CE (公式9)
    # L_CE ← (softmax(φ_m(h_i)), y_i)
    loss_ce = ctx.criterion(pred, label)

    if ctx.cur_mode == MODE.TRAIN:
        # Step 4: 计算紧凑性损失 L_COM (公式8)
        # L_COM ← ({a_j}_{j=1}^C, h_i)
        loss_com = self.compute_compactness_loss(h, label)

        # Step 5: 总损失 L_HC (公式10)
        # L_HC = L_CE + λ·L_COM
        total_loss = loss_ce + self.lambda_com * loss_com

        ctx.loss_batch = CtxVar(total_loss, LIFECYCLE.BATCH)
        ctx.loss_ce = CtxVar(loss_ce, LIFECYCLE.BATCH)
        ctx.loss_com = CtxVar(loss_com, LIFECYCLE.BATCH)

    # Step 6: 反向传播 (自动由框架处理)
    # Φ_m^e ← Φ_m^{e-1} - η∇L_HC
```

### 4.4 嵌入提取机制

```python
# trainer_FedLSA.py:121
def _forward_with_embedding(self, model, x):
    """
    从模型中提取中间嵌入 z (在最终分类器之前)

    支持的模型架构:
    1. FedLSACNN: conv1 → conv2 → fc1(z) → fc2 → fc3(q)
    2. ConvNet2: conv1 → conv2 → fc1(z) → fc2(q)

    Args:
        model: 神经网络模型
        x: 输入张量 [B, C, H, W]

    Returns:
        pred: 模型预测 (logits) [B, num_classes]
        z: 特征嵌入 [B, hidden_dim]
    """
    # 检测3层全连接的架构 (FedLSACNN)
    if hasattr(model, 'fc3') and hasattr(model, 'fc2') and hasattr(model, 'fc1'):
        z = x

        # 卷积层
        z = model.conv1(z)
        if model.use_bn:
            z = model.bn1(z)
        z = model.relu(z)
        z = model.maxpool(z)

        z = model.conv2(z)
        if model.use_bn:
            z = model.bn2(z)
        z = model.relu(z)
        z = model.maxpool(z)

        # 展平
        z = z.view(z.size(0), -1)

        # FC1: 得到嵌入 z
        if hasattr(model, 'dropout'):
            z = F.dropout(z, p=model.dropout, training=model.training)
        embedding = model.relu(model.fc1(z))  # 这是我们的嵌入 z

        # 继续前向到输出
        z = embedding
        if hasattr(model, 'dropout'):
            z = F.dropout(z, p=model.dropout, training=model.training)
        z = model.relu(model.fc2(z))

        if hasattr(model, 'dropout'):
            z = F.dropout(z, p=model.dropout, training=model.training)
        pred = model.fc3(z)

        return pred, embedding

    # 类似处理2层全连接架构...
```

### 4.5 消息传递机制

**服务器广播锚点**:

```python
# server_FedLSA.py:305
def broadcast_semantic_anchors(self, receiver=None):
    """
    向客户端广播语义锚点

    消息格式:
        msg_type: 'semantic_anchors'
        content: semantic_anchors张量 [C, D]
    """
    if receiver is None:
        receiver = list(self.comm_manager.neighbors.keys())

    # 发送锚点消息
    self.comm_manager.send(
        Message(msg_type='semantic_anchors',
                sender=self.ID,
                receiver=receiver,
                state=self.state,
                timestamp=self.cur_timestamp,
                content=self.semantic_anchors.detach().cpu())
    )
```

**客户端接收锚点**:

```python
# client_FedLSA.py:48
def callback_funcs_for_semantic_anchors(self, message: Message):
    """
    处理从服务器接收的语义锚点

    Args:
        message: 包含semantic_anchors的消息对象
    """
    content = message.content  # 语义锚点张量 [C, D]

    if content is not None:
        # 更新Trainer中的语义锚点
        if hasattr(self.trainer, 'update_semantic_anchors'):
            self.trainer.update_semantic_anchors(content)
        else:
            logger.warning("Trainer不支持锚点更新")
```

---

## 配置系统

### 5.1 配置文件结构

**FedLSA配置** (`federatedscope/core/configs/cfg_fedlsa.py`):

```yaml
# FedLSA 特定配置
fedlsa:
  use: True                          # 是否使用FedLSA

  # 客户端超参数
  lambda_com: 0.5                    # λ: 紧凑性损失权重 (公式10)
  tau: 0.1                           # τ: 温度参数 (公式7,8)
  use_projector: True                # 是否使用超球面投影器
  projector_input_dim: 512           # 投影器输入维度 (I)
  projector_output_dim: 128          # 投影器输出维度 (L)
  share_projector: True              # 是否共享投影器参数

  # 服务器超参数
  alpha_sep: 0.4                     # α: 分离损失权重 (公式5)
  anchor_train_epochs: 500           # E_s: 锚点训练轮数
  anchor_lr: 0.001                   # η: 锚点学习率

# 模型配置
model:
  type: 'fedlsa_cnn'                 # 模型类型
  num_classes: 10                    # C: 类别数
  hidden: 512                        # I: 嵌入维度

# 训练配置
trainer:
  type: 'fedlsa_trainer'             # 使用FedLSA Trainer

# 联邦学习配置
federate:
  method: 'fedlsa'                   # 联邦方法
  total_round_num: 100               # T: 通信轮数
  client_num: 10                     # M: 客户端数量

train:
  local_update_steps: 5              # E_c: 本地训练轮数
  batch_size: 64
  optimizer:
    type: SGD
    lr: 0.01                         # 本地学习率
```

### 5.2 完整配置示例

```yaml
# scripts/example_configs/fedlsa_office_caltech.yaml
use_gpu: True
device: 0
early_stop:
  patience: 50

data:
  root: 'data/'
  type: 'office_caltech@torchvision'  # 使用Office Caltech数据集
  splits: [0.8, 0.1, 0.1]
  subsample: 1.0
  batch_size: 64

model:
  type: 'fedlsa_cnn'
  num_classes: 10
  hidden: 512
  dropout: 0.1

train:
  optimizer:
    type: SGD
    lr: 0.01
    momentum: 0.9
    weight_decay: 0.0005
  local_update_steps: 5

federate:
  mode: 'standalone'
  method: 'fedlsa'
  total_round_num: 100
  client_num: 10
  sample_client_num: 10

fedlsa:
  use: True
  # 客户端参数
  lambda_com: 0.5      # 论文推荐: {0.1, 0.7}
  tau: 0.1             # 论文推荐: {0.1, 0.2}
  use_projector: True
  projector_input_dim: 512
  projector_output_dim: 128
  share_projector: True
  # 服务器参数
  alpha_sep: 0.4       # 论文推荐: {0.02, 0.4}
  anchor_train_epochs: 500
  anchor_lr: 0.001

trainer:
  type: 'fedlsa_trainer'

eval:
  freq: 1
  metrics: ['acc', 'loss']
```

---

## 完整训练流程

### 6.1 训练流程图

```
┌─────────────────────────────────────────────────────────────────────┐
│                     FedLSA Training Process                          │
└─────────────────────────────────────────────────────────────────────┘

初始化阶段:
  Server: 初始化 R, Θ, φ_glo, Φ_glo
  Clients: 初始化 Φ_m, φ_m

通信轮次 t = 1 to T:

  ┌─────────────────────────────────────────────────────────┐
  │ Server Side (Round t)                                   │
  ├─────────────────────────────────────────────────────────┤
  │ 1. 训练语义锚点 (E_s=500轮):                              │
  │    for e = 1 to E_s:                                    │
  │      A = Θ(R)                                           │
  │      L_ACE = -1_{y_i} log(softmax(φ_glo(a_i)))          │
  │      L_SEP = log(Σ_{j≠i} exp(a_i·a_j^T / τ) / (C-1))   │
  │      L_LSA = L_ACE + α·L_SEP                            │
  │      R, Θ, φ_glo ← 优化 L_LSA                           │
  │                                                         │
  │ 2. 聚合客户端模型 (FedAvg):                              │
  │    Φ_glo = Σ_m (n_m/n)·Φ_m                              │
  │                                                         │
  │ 3. 广播到客户端:                                         │
  │    broadcast(A, Φ_glo) → Clients                        │
  └─────────────────────────────────────────────────────────┘
                            ↓
  ┌─────────────────────────────────────────────────────────┐
  │ Client Side (Round t, Client m, m=1...M in parallel)   │
  ├─────────────────────────────────────────────────────────┤
  │ 1. 接收锚点和全局模型:                                    │
  │    receive(A, Φ_glo)                                    │
  │    Φ_m ← Φ_glo                                          │
  │                                                         │
  │ 2. 本地训练 (E_c=5轮):                                   │
  │    for e = 1 to E_c:                                    │
  │      for batch (x_i, y_i) in D_m:                       │
  │        # 前向传播                                        │
  │        z_i = ψ_m(x_i)           # 编码                   │
  │        h_i = nor(φ_m(z_i))      # 超球面投影             │
  │        q_i = φ_m(h_i)           # 分类                   │
  │                                                         │
  │        # 损失计算                                        │
  │        L_CE = -1_{y_i} log(softmax(q_i))      (公式9)    │
  │        L_COM = -log(exp(a_{y_i}^T·h_i/τ) /             │
  │                     Σ_j exp(a_j^T·h_i/τ))     (公式8)    │
  │        L_HC = L_CE + λ·L_COM                  (公式10)   │
  │                                                         │
  │        # 反向传播                                        │
  │        Φ_m ← Φ_m - η∇L_HC                               │
  │                                                         │
  │ 3. 上传本地模型:                                         │
  │    upload(Φ_m) → Server                                 │
  └─────────────────────────────────────────────────────────┘
                            ↓
              (返回Server Side, t = t+1)

最终输出: 全局模型 Φ_glo
```

### 6.2 算法伪代码对照

**论文算法1完整版**:

```
Algorithm 1: The flow of FedLSA
Input:
  - M: 客户端数量
  - E_c, E_s: 客户端和服务器训练轮数
  - T: 通信轮数
  - η: 学习率
  - τ: 温度参数
  - D_m = {(x_i, y_i)}_{i=1}^{|D_m|}: 本地数据
  - R: 锚点初始化参数

Output:
  - Φ_glo: 全局模型

=== Client Side ===
for m = 1:M in parallel do
  for epoch e = 1:E_c do
    h_i = nor(φ_m(ψ_m(x_i)))

    // 计算CE损失 L_CE (公式9)
    L_CE ← (softmax(φ_m(h_i)), y_i)

    // 计算紧凑性损失 L_COM (公式8)
    L_COM ← ({a_j}_{j=1}^C, h_i)

    L_HC = L_CE + λ·L_COM
    Φ_m^e ← Φ_m^{e-1} - η∇L_HC
  end for
end for

Upload Φ_m → Server

=== Server Side ===
for t = 1:T do
  for e = 1:E_s do
    A = Θ(R)

    // 计算锚点CE损失 L_ACE (公式3)
    L_ACE ← (softmax(φ_glo(a^i)), y_i)

    // 计算分离损失 L_SEP (公式4)
    L_SEP ← ({a_j}_{j=1}^C)

    L_LSA = L_ACE + α·L_SEP
    R^e ← R^{e-1} - η∇L_LSA
    Θ^e ← Θ^{e-1} - η∇L_LSA
  end for

  Broadcast A, Φ_glo → Clients
end for
```

### 6.3 代码执行流程

**启动脚本** (`scripts/run_fedlsa.py`):

```python
def main():
    # 1. 初始化配置
    cfg = get_config('scripts/example_configs/fedlsa_office_caltech.yaml')

    # 2. 设置随机种子
    set_random_seed(cfg.seed)

    # 3. 创建数据加载器
    data, modified_config = get_data(cfg.clone())

    # 4. 创建模型
    model = get_model(modified_config)

    # 5. 创建Server和Clients
    server = FedLSAServer(
        ID=0,
        config=modified_config,
        model=model,
        client_num=cfg.federate.client_num,
        total_round_num=cfg.federate.total_round_num,
        device=cfg.device
    )

    clients = []
    for client_id in range(1, cfg.federate.client_num + 1):
        client = FedLSAClient(
            ID=client_id,
            server_id=0,
            config=modified_config,
            data=data[client_id],
            model=copy.deepcopy(model),
            device=cfg.device
        )
        clients.append(client)

    # 6. 运行联邦学习
    server.run()

if __name__ == '__main__':
    main()
```

**单轮次详细流程**:

```python
# 伪代码展示一个通信轮次的完整流程

def one_communication_round(server, clients, round_num):
    """
    单个通信轮次的完整流程

    Args:
        server: FedLSAServer实例
        clients: FedLSAClient列表
        round_num: 当前轮次
    """
    print(f"\n=== Round {round_num} ===")

    # ========== 服务器端 ==========
    print("[Server] Training semantic anchors...")
    server.train_semantic_anchors()
    # 输出: L_ACE, L_SEP, L_LSA

    print("[Server] Broadcasting anchors and model...")
    server.broadcast_semantic_anchors()
    server.broadcast_model_para()

    # ========== 客户端并行训练 ==========
    client_models = []
    for client in clients:
        print(f"[Client {client.ID}] Receiving anchors and model...")
        anchors = server.semantic_anchors
        model_para = server.get_model_para()

        client.trainer.update_semantic_anchors(anchors)
        client.trainer.update(model_para)

        print(f"[Client {client.ID}] Local training...")
        for epoch in range(cfg.train.local_update_steps):  # E_c=5
            for batch_idx, (x, y) in enumerate(client.data.train_loader):
                # 前向传播 (算法1 Client Side)
                loss_dict = client.trainer.train_step(x, y)
                # 包含: loss_batch, loss_ce, loss_com

                # 反向传播由trainer自动处理

        print(f"[Client {client.ID}] Uploading model...")
        client_models.append(client.get_model_para())

    # ========== 服务器端聚合 ==========
    print("[Server] Aggregating client models...")
    server.aggregate(client_models)

    # ========== 评估 ==========
    if round_num % cfg.eval.freq == 0:
        print("[Server] Evaluating global model...")
        metrics = server.evaluate()
        print(f"Round {round_num} - Acc: {metrics['acc']:.4f}, Loss: {metrics['loss']:.4f}")
```

### 6.4 关键时序

```
Time     Server                           Clients
────────────────────────────────────────────────────────────
t=0      初始化R,Θ,φ_glo,Φ_glo             初始化Φ_m,φ_m
         │
         ↓
         训练锚点 (500轮)
         A = Θ(R)                          (等待)
         优化 L_LSA
         │
         ├─── broadcast(A,Φ_glo) ─────────→ 接收 A,Φ_glo
         │                                 │
         ↓                                 ↓
         (等待)                            本地训练 (5轮)
                                           优化 L_HC
                                           │
         ←────── upload(Φ_m) ─────────────┤
         │
         ↓
         聚合模型
         Φ_glo = avg(Φ_m)
         │
         ↓
t=1      训练锚点 (500轮)
         ...                               ...

(重复T=100轮)
```

---

## 公式与代码对应表

| 公式编号 | 数学表达 | 代码位置 | 函数/方法 |
|---------|---------|---------|----------|
| 公式1 | $p_m^c = \frac{1}{\|D_m^c\|}\sum_{i}\psi_m(x_i)$ | N/A | 传统方法(FedLSA不使用) |
| 公式2 | $p^c = \frac{1}{M}\sum_{m}p_m^c$ | N/A | 传统方法(FedLSA不使用) |
| 公式3 | $\mathcal{L}_{ACE} = -\mathbf{1}_{y_i}\log(\text{softmax}(\rho_i))$ | `server_FedLSA.py:124` | `compute_anchor_ce_loss()` |
| 公式4 | $\mathcal{L}_{SEP} = \log\frac{\sum_{j\neq i}\exp(a_ia_j^T/\tau)}{C-1}$ | `server_FedLSA.py:147` | `compute_separation_loss()` |
| 公式5 | $\mathcal{L}_{LSA} = \mathcal{L}_{ACE} + \alpha\mathcal{L}_{SEP}$ | `server_FedLSA.py:192` | `train_semantic_anchors()` |
| 公式6 | $p_D(h;a_c,\kappa) = N_D(\kappa)\exp(\kappa a_c^Th)$ | N/A | 理论基础 (von Mises-Fisher) |
| 公式7 | $p(y_i=c\|h_i) = \frac{\exp(a_c^Th_i/\tau)}{\sum_j\exp(a_j^Th_i/\tau)}$ | `trainer_FedLSA.py:95` | `compute_compactness_loss()` |
| 公式8 | $\mathcal{L}_{COM} = -\log\frac{\exp(a_{y_i}^Th_i/\tau)}{\sum_j\exp(a_j^Th_i/\tau)}$ | `trainer_FedLSA.py:95` | `compute_compactness_loss()` |
| 公式9 | $\mathcal{L}_{CE} = -\mathbf{1}_{y_i}\log(\text{softmax}(q_i))$ | `trainer_FedLSA.py:228` | `_hook_on_batch_forward()` |
| 公式10 | $\mathcal{L}_{HC} = \mathcal{L}_{CE} + \lambda\mathcal{L}_{COM}$ | `trainer_FedLSA.py:228` | `_hook_on_batch_forward()` |

---

## 使用示例

### 7.1 快速开始

```bash
# 1. 下载数据集
python scripts/download_office_caltech.py

# 2. 运行FedLSA
python scripts/run_fedlsa.py \
    --cfg scripts/example_configs/fedlsa_office_caltech.yaml

# 3. 查看结果
# 输出会显示每轮的准确率和损失
```

### 7.2 自定义配置

```python
# custom_fedlsa_config.yaml
fedlsa:
  # 调整超参数
  lambda_com: 0.7        # 增大紧凑性损失权重
  alpha_sep: 0.02        # 减小分离损失权重
  tau: 0.2               # 增大温度参数

  # 修改锚点训练
  anchor_train_epochs: 300  # 减少训练轮数以加速
  anchor_lr: 0.005          # 增大学习率

model:
  hidden: 256            # 减小嵌入维度
  dropout: 0.2           # 增大Dropout

fedlsa:
  projector_output_dim: 64  # 减小超球面维度
```

### 7.3 扩展到新数据集

```python
# 1. 创建数据集类 (如 federatedscope/cv/dataset/my_dataset.py)
class MyDataset:
    def __init__(self, root, client_id):
        # 加载数据
        self.data = load_my_data(root, client_id)

    def __getitem__(self, idx):
        return self.data[idx]

# 2. 注册数据集
from federatedscope.register import register_data
register_data('my_dataset', MyDataset)

# 3. 修改配置
# my_dataset_config.yaml
data:
  type: 'my_dataset'
  root: 'data/my_dataset/'

model:
  num_classes: 20        # 根据你的数据集调整
```

---

## 实验结果复现

### 8.1 论文报告的性能

**Office Caltech数据集**:
- FedAvg: 53.20%
- FedProto: 52.01%
- FedLSA: **60.22%** (+7.02%)

**Digits数据集**:
- FedAvg: 60.89%
- FedProto: 61.20%
- FedLSA: **63.56%** (+2.67%)

**PACS数据集**:
- FedAvg: 40.36%
- FedProto: 42.07%
- FedLSA: **43.73%** (+3.37%)

### 8.2 消融研究

| L_SEP | L_COM | Digits Avg | Office Caltech Avg |
|-------|-------|------------|-------------------|
| ✘     | ✘     | 60.89      | 53.20             |
| ✘     | ✓     | 61.06      | 59.18             |
| ✓     | ✘     | 61.21      | 58.37             |
| ✓     | ✓     | **63.56**  | **60.22**         |

**结论**: 两个损失都是必需的,缺一不可。

---

## 常见问题

### Q1: 为什么要用超球面空间而不是欧几里得空间?
**A**: 超球面空间确保嵌入均匀分布,避免特征坍塌。论文图2显示FedProto在欧氏空间的嵌入极不均匀,甚至有点跳出超球面,而FedLSA在超球面上分布均匀。

### Q2: 语义锚点和传统原型有什么区别?
**A**:
- **传统原型**: 通过类内样本平均得到,携带域特定信息
- **语义锚点**: 通过网络训练学习,只与类别语义相关,独立于域

### Q3: 温度参数τ的作用是什么?
**A**: 控制对比学习的集中度。较小的τ使分布更尖锐(高置信度),较大的τ使分布更平滑(低置信度)。论文推荐τ=0.1或0.2。

### Q4: 为什么服务器要训练500轮锚点?
**A**: 锚点需要充分优化以确保:
1. 每个锚点对应正确的语义类别 (L_ACE)
2. 锚点间有足够的分离度 (L_SEP)

### Q5: 如何调整超参数?
**A**: 论文推荐范围:
- α (alpha_sep): {0.02, 0.4}
- λ (lambda_com): {0.1, 0.7}
- τ (tau): {0.1, 0.2}

从中间值开始,根据验证集性能调整。

---

## 总结

FedLSA通过以下创新解决域偏斜问题:

1. **语义锚点学习**: 直接学习与域无关的类别表示
2. **超球面对比学习**: 确保嵌入均匀分布
3. **双重优化目标**:
   - 服务器: 最大化类间分离 (L_LSA)
   - 客户端: 最大化类内紧凑 (L_HC)

**核心公式链**:
```
服务器: A = Θ(R) → L_LSA = L_ACE + α·L_SEP → 优化R,Θ
客户端: h = nor(φ(ψ(x))) → L_HC = L_CE + λ·L_COM → 优化Φ
```

**代码结构清晰**:
- `SemanticAnchorLearner`: 锚点生成
- `FedLSAServer`: 锚点训练与分发
- `FedLSAClient`: 锚点接收
- `FedLSATrainer`: 超球面投影与紧凑性学习
- `FedLSACNN`: 特征提取网络

通过本文档,你应该能够:
1. 理解FedLSA的理论基础和数学原理
2. 掌握代码实现细节和关键函数
3. 配置和运行FedLSA实验
4. 扩展FedLSA到新场景

---

**参考文献**:
Fu, L., Huang, S., Lai, Y., Liao, T., Zhang, C., & Chen, C. (2025). Beyond Federated Prototype Learning: Learnable Semantic Anchors with Hyperspherical Contrast for Domain-Skewed Data. *AAAI Conference on Artificial Intelligence*.
