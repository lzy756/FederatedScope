# GGEUR 方法在 FederatedScope 框架下的复现工作纲要

## 论文信息
- **论文名称**: Geometric Knowledge-Guided Localized Global Distribution Alignment for Federated Learning
- **会议**: CVPR 2025
- **核心方法**: GGEUR (Global Geometry-Guided Embedding Uncertainty Representation)
- **原始代码**: https://github.com/WeiDai-David/2025CVPR_GGEUR

---

## 一、方法核心理解

### 1.1 核心思想
GGEUR 通过**几何知识指导的数据增强**来缓解联邦学习中的数据异构问题，使本地数据分布更接近全局分布。

### 1.2 技术路线

#### 单域场景（处理标签偏斜）
1. 客户端使用 CLIP 提取本地数据的嵌入特征
2. 客户端计算每个类别的局部协方差矩阵 Σ_k^i 和均值 μ_k^i
3. 服务器聚合所有客户端的协方差矩阵，得到全局协方差矩阵 Σ_i
4. 服务器对 Σ_i 进行特征值分解，得到全局几何形状 GD_i = {ξ_1,...,ξ_p, λ_1,...,λ_p}
5. 服务器下发全局几何形状给客户端
6. 客户端使用 GGEUR 算法生成新样本：X_new = X + Σ(ε_m * λ_m * ξ_m)
7. 客户端在增强后的数据上训练 MLP 分类器

#### 多域场景（处理标签偏斜 + 域偏斜）
**Step 1**: 本地域增强
- 与单域场景相同，使用全局几何形状增强本地样本

**Step 2**: 跨域模拟
- 服务器收集其他域的类原型（均值）
- 服务器下发其他域的原型给客户端
- 客户端对每个其他域的原型应用 GGEUR，生成跨域样本

---

## 二、实现架构设计

### 2.1 模块结构

```
federatedscope/
├── contrib/
│   ├── worker/
│   │   ├── ggeur_client.py          # GGEUR 客户端（新建）
│   │   ├── ggeur_server.py          # GGEUR 服务器（新建）
│   ├── trainer/
│   │   ├── ggeur_trainer.py         # GGEUR 训练器（新建）
│   ├── data_augmentation/
│   │   ├── ggeur_augmentation.py    # GGEUR 数据增强核心算法（新建）
│   ├── utils/
│   │   ├── clip_extractor.py        # CLIP 特征提取器（新建）
│   │   ├── geometry_utils.py        # 几何形状计算工具（新建）
│   │   ├── covariance_aggregator.py # 协方差矩阵聚合（新建）
├── core/
│   ├── configs/
│   │   ├── cfg_ggeur.py            # GGEUR 配置（新建）
scripts/
├── example_configs/
│   ├── ggeur_cifar10_label_skew.yaml    # CIFAR-10 标签偏斜（新建）
│   ├── ggeur_cifar100_label_skew.yaml   # CIFAR-100 标签偏斜（新建）
│   ├── ggeur_digits_domain_skew.yaml    # Digits 域偏斜（新建）
│   ├── ggeur_office_home_lds.yaml       # Office-Home 混合偏斜（新建）
```

### 2.2 核心类设计

#### GGEURClient (继承自 FedLSAClient)
```python
class GGEURClient(FedLSAClient):
    """GGEUR_Clip 客户端"""

    职责：
    1. 提取本地数据的 CLIP 嵌入
    2. 计算每个类别的局部协方差矩阵和均值
    3. 上传协方差矩阵和均值到服务器
    4. 接收全局几何形状和其他域原型
    5. 使用 GGEUR 进行数据增强
    6. 训练本地 MLP 模型

    新增属性：
    - clip_model: CLIP 模型实例
    - local_embeddings: 本地嵌入特征缓存
    - local_cov_matrices: 局部协方差矩阵 {class_id: Σ_k^i}
    - local_means: 局部均值 {class_id: μ_k^i}
    - global_geometries: 全局几何形状 {class_id: (eigenvalues, eigenvectors)}
    - other_domain_prototypes: 其他域的类原型
    - augmented_data: 增强后的数据集

    新增方法：
    - extract_embeddings(): 使用 CLIP 提取嵌入
    - compute_local_statistics(): 计算局部协方差矩阵和均值
    - callback_for_global_geometries(): 接收全局几何形状
    - callback_for_domain_prototypes(): 接收域原型
    - augment_data_ggeur(): 执行 GGEUR 数据增强
```

#### GGEURServer (继承自 FedLSAServer)
```python
class GGEURServer(FedLSAServer):
    """GGEUR_Clip 服务器"""

    职责：
    1. 收集所有客户端的局部协方差矩阵和均值
    2. 聚合计算全局协方差矩阵
    3. 对全局协方差矩阵进行特征值分解
    4. 下发全局几何形状给客户端
    5. 收集和分发域原型（多域场景）

    新增属性：
    - local_cov_buffer: 缓存客户端上传的协方差矩阵
    - local_mean_buffer: 缓存客户端上传的均值
    - global_cov_matrices: 全局协方差矩阵 {class_id: Σ_i}
    - global_geometries: 全局几何形状
    - domain_prototypes: 域原型集合
    - scenario: 'single_domain' 或 'multi_domain'

    新增方法：
    - aggregate_covariances(): 聚合协方差矩阵（公式4）
    - compute_global_geometries(): 计算全局几何形状（特征值分解）
    - broadcast_global_geometries(): 下发全局几何形状
    - collect_domain_prototypes(): 收集域原型
    - broadcast_domain_prototypes(): 下发域原型
```

#### GGEURTrainer (继承自 Trainer)
```python
class GGEURTrainer(Trainer):
    """GGEUR_Clip 训练器"""

    职责：
    1. 在增强后的嵌入数据上训练 MLP
    2. 支持单域和多域场景

    新增属性：
    - augmented_embeddings: 增强后的嵌入特征
    - augmented_labels: 对应的标签
    - mlp_classifier: MLP 分类器

    新增方法：
    - setup_augmented_data(): 设置增强数据
    - train_on_embeddings(): 在嵌入上训练
```

---

## 三、详细实现步骤

### 阶段 1: 基础设施搭建（第 1-2 周）

#### Task 1.1: CLIP 特征提取器
**文件**: `federatedscope/contrib/utils/clip_extractor.py`

**功能**:
- 加载预训练的 CLIP 模型（ViT-B/16）
- 提取图像嵌入特征
- 支持批量处理
- 缓存机制（避免重复提取）

**关键接口**:
```python
class CLIPExtractor:
    def __init__(self, model_name='ViT-B/16', device='cuda'):
        """初始化 CLIP 模型"""

    def extract_features(self, images):
        """提取图像特征，返回 [N, D] 嵌入"""

    def extract_dataset_features(self, dataset):
        """提取整个数据集的特征"""
```

#### Task 1.2: 几何形状计算工具
**文件**: `federatedscope/contrib/utils/geometry_utils.py`

**功能**:
- 计算协方差矩阵
- 特征值分解
- 几何形状相似度计算（可选，用于验证）

**关键接口**:
```python
def compute_covariance_matrix(embeddings, mean=None):
    """
    计算协方差矩阵
    Args:
        embeddings: [N, D] 张量
        mean: 均值向量（可选）
    Returns:
        covariance: [D, D] 协方差矩阵
    """

def eigendecompose(covariance_matrix, top_k=None):
    """
    特征值分解
    Args:
        covariance_matrix: [D, D] 协方差矩阵
        top_k: 保留前 k 个特征值（可选）
    Returns:
        eigenvalues: [K] 特征值
        eigenvectors: [D, K] 特征向量
    """

def compute_geometry_similarity(geometry1, geometry2):
    """计算几何形状相似度（公式3）"""
```

#### Task 1.3: 协方差矩阵聚合器
**文件**: `federatedscope/contrib/utils/covariance_aggregator.py`

**功能**:
- 实现公式 (4) 的协方差矩阵聚合

**关键接口**:
```python
def aggregate_covariances(local_covs, local_means, sample_counts):
    """
    聚合局部协方差矩阵得到全局协方差矩阵

    公式4实现:
    Σ_i = 1/N_i * (Σ n_k^i * Σ_k^i + Σ n_k^i * (μ_k^i - μ_i)(μ_k^i - μ_i)^T)

    Args:
        local_covs: 列表，每个元素是 [D, D] 局部协方差矩阵
        local_means: 列表，每个元素是 [D] 局部均值
        sample_counts: 列表，每个元素是样本数 n_k^i
    Returns:
        global_cov: [D, D] 全局协方差矩阵
        global_mean: [D] 全局均值
    """
```

---

### 阶段 2: GGEUR 数据增强算法（第 3 周）

#### Task 2.1: 单域 GGEUR 实现
**文件**: `federatedscope/contrib/data_augmentation/ggeur_augmentation.py`

**功能**:
- 实现 Algorithm 1（单域场景）
- 基于全局几何形状生成新样本

**关键接口**:
```python
class GGEURAugmentation:
    def __init__(self, global_geometries, scenario='single_domain'):
        """
        Args:
            global_geometries: {class_id: {'eigenvalues': λ, 'eigenvectors': ξ}}
            scenario: 'single_domain' 或 'multi_domain'
        """

    def augment_single_domain(self, embeddings, labels, class_id, N=10):
        """
        单域 GGEUR_Clip 增强（Algorithm 1）

        Args:
            embeddings: [n_k^i, D] 局部类别嵌入
            labels: [n_k^i] 标签
            class_id: 类别 ID
            N: 每个样本生成的新样本数
        Returns:
            augmented_embeddings: [n_k^i * N, D]
            augmented_labels: [n_k^i * N]
        """
        # 1. 获取该类别的全局几何形状
        geometry = self.global_geometries[class_id]
        eigenvalues = geometry['eigenvalues']  # [p]
        eigenvectors = geometry['eigenvectors']  # [D, p]

        # 2. 为每个原始样本生成 N 个新样本
        new_samples = []
        for x in embeddings:
            for _ in range(N):
                # 生成随机向量 β = Σ ε_m * λ_m * ξ_m
                epsilon = torch.randn(len(eigenvalues))
                beta = torch.sum(epsilon * eigenvalues * eigenvectors.T, dim=0)

                # 生成新样本 X_new = X + β
                x_new = x + beta
                new_samples.append(x_new)

        return torch.stack(new_samples), labels.repeat(N)
```

#### Task 2.2: 多域 GGEUR 实现
**扩展上述类**:

**功能**:
- 实现 Algorithm 2（多域场景）
- Step 1: 本地域增强
- Step 2: 跨域样本生成

**关键方法**:
```python
    def augment_multi_domain(self,
                            embeddings,
                            labels,
                            class_id,
                            other_domain_prototypes,
                            N=10,
                            M=500):
        """
        多域 GGEUR_Clip 增强（Algorithm 2）

        Args:
            embeddings: [n_k^i, D] 局部嵌入
            labels: [n_k^i] 标签
            class_id: 类别 ID
            other_domain_prototypes: {domain_id: prototype} 其他域的原型
            N: Step 1 每个样本生成数
            M: Step 2 每个原型生成数
        Returns:
            augmented_embeddings: 增强后的嵌入
            augmented_labels: 对应标签
        """
        # Step 1: 本地域增强
        step1_embeddings, step1_labels = self.augment_single_domain(
            embeddings, labels, class_id, N
        )

        # Step 2: 跨域模拟
        step2_embeddings = []
        step2_labels = []

        for domain_id, prototype in other_domain_prototypes.items():
            # 对每个其他域的原型应用 GGEUR_Clip
            for _ in range(M):
                epsilon = torch.randn(len(eigenvalues))
                beta = torch.sum(epsilon * eigenvalues * eigenvectors.T, dim=0)
                x_new = prototype + beta
                step2_embeddings.append(x_new)
                step2_labels.append(class_id)

        # 合并 Step 1 和 Step 2 的结果
        all_embeddings = torch.cat([step1_embeddings,
                                    torch.stack(step2_embeddings)])
        all_labels = torch.cat([step1_labels,
                               torch.tensor(step2_labels)])

        return all_embeddings, all_labels
```

---

### 阶段 3: 客户端实现（第 4 周）

#### Task 3.1: GGEURClient 实现
**文件**: `federatedscope/contrib/worker/ggeur_client.py`

**关键流程**:
```python
class GGEURClient(FedLSAClient):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # 初始化 CLIP 提取器
        self.clip_extractor = CLIPExtractor(
            model_name=self._cfg.ggeur.clip_model,
            device=self.device
        )

        # 数据结构
        self.local_embeddings = None  # 缓存嵌入
        self.local_cov_matrices = {}  # {class_id: Σ_k^i}
        self.local_means = {}  # {class_id: μ_k^i}
        self.global_geometries = None
        self.other_domain_prototypes = None
        self.augmented_data = None

        # 注册消息处理器
        self.register_handlers('global_geometries',
                              self.callback_for_global_geometries)
        self.register_handlers('domain_prototypes',
                              self.callback_for_domain_prototypes)

    def _extract_and_compute_statistics(self):
        """阶段1: 提取嵌入并计算统计量"""
        logger.info(f"Client #{self.ID}: 提取 CLIP 嵌入...")

        # 提取训练数据的嵌入
        train_data = self.data['train']
        self.local_embeddings = self.clip_extractor.extract_dataset_features(
            train_data
        )

        # 按类别计算协方差矩阵和均值
        logger.info(f"Client #{self.ID}: 计算局部统计量...")
        unique_classes = torch.unique(train_data.targets)

        for class_id in unique_classes:
            # 获取该类别的嵌入
            class_mask = (train_data.targets == class_id)
            class_embeddings = self.local_embeddings[class_mask]

            # 计算均值
            mean = class_embeddings.mean(dim=0)
            self.local_means[class_id.item()] = mean

            # 计算协方差矩阵
            cov = compute_covariance_matrix(class_embeddings, mean)
            self.local_cov_matrices[class_id.item()] = cov

        # 上传到服务器
        self._upload_statistics()

    def _upload_statistics(self):
        """上传局部统计量到服务器"""
        content = {
            'cov_matrices': self.local_cov_matrices,
            'means': self.local_means,
            'sample_counts': {
                cid: (self.data['train'].targets == cid).sum().item()
                for cid in self.local_cov_matrices.keys()
            }
        }

        self.comm_manager.send(
            Message(
                msg_type='local_statistics',
                sender=self.ID,
                receiver=[self.server_id],
                state=self.state,
                content=content
            )
        )

    def callback_for_global_geometries(self, message):
        """接收全局几何形状"""
        self.global_geometries = message.content
        logger.info(f"Client #{self.ID}: 收到全局几何形状")

        # 触发数据增强
        self._perform_augmentation()

    def callback_for_domain_prototypes(self, message):
        """接收域原型（多域场景）"""
        self.other_domain_prototypes = message.content
        logger.info(f"Client #{self.ID}: 收到域原型")

    def _perform_augmentation(self):
        """执行 GGEUR_Clip 数据增强"""
        logger.info(f"Client #{self.ID}: 开始 GGEUR_Clip 数据增强...")

        augmenter = GGEURAugmentation(
            global_geometries=self.global_geometries,
            scenario=self._cfg.ggeur.scenario
        )

        augmented_embeddings = []
        augmented_labels = []

        for class_id in self.local_cov_matrices.keys():
            # 获取该类别的原始嵌入
            class_mask = (self.data['train'].targets == class_id)
            class_embeddings = self.local_embeddings[class_mask]
            class_labels = torch.full((len(class_embeddings),), class_id)

            if self._cfg.ggeur.scenario == 'single_domain':
                # 单域增强
                aug_emb, aug_lab = augmenter.augment_single_domain(
                    class_embeddings,
                    class_labels,
                    class_id,
                    N=self._cfg.ggeur.n_samples_per_original
                )
            else:
                # 多域增强
                other_prototypes = {
                    k: v for k, v in self.other_domain_prototypes.items()
                    if k != self.ID  # 排除自己的域
                }
                aug_emb, aug_lab = augmenter.augment_multi_domain(
                    class_embeddings,
                    class_labels,
                    class_id,
                    other_prototypes.get(class_id, {}),
                    N=self._cfg.ggeur.n_samples_per_original,
                    M=self._cfg.ggeur.m_samples_per_prototype
                )

            augmented_embeddings.append(aug_emb)
            augmented_labels.append(aug_lab)

        # 合并所有类别的增强数据
        self.augmented_data = {
            'embeddings': torch.cat(augmented_embeddings),
            'labels': torch.cat(augmented_labels)
        }

        logger.info(f"Client #{self.ID}: 增强完成，"
                   f"原始样本: {len(self.local_embeddings)}, "
                   f"增强后: {len(self.augmented_data['embeddings'])}")

        # 更新 trainer 的数据
        self.trainer.setup_augmented_data(self.augmented_data)
```

---

### 阶段 4: 服务器实现（第 5 周）

#### Task 4.1: GGEURServer 实现
**文件**: `federatedscope/contrib/worker/ggeur_server.py`

**关键流程**:
```python
class GGEURServer(FedLSAServer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # 缓冲区
        self.local_cov_buffer = {}  # {client_id: {class_id: Σ_k^i}}
        self.local_mean_buffer = {}  # {client_id: {class_id: μ_k^i}}
        self.sample_count_buffer = {}  # {client_id: {class_id: n_k^i}}

        # 全局数据
        self.global_cov_matrices = {}  # {class_id: Σ_i}
        self.global_geometries = {}  # {class_id: (λ, ξ)}
        self.domain_prototypes = {}  # {client_id: {class_id: prototype}}

        # 注册消息处理器
        self.register_handlers('local_statistics',
                              self.callback_for_local_statistics)

    def callback_for_local_statistics(self, message):
        """接收客户端的局部统计量"""
        client_id = message.sender
        content = message.content

        self.local_cov_buffer[client_id] = content['cov_matrices']
        self.local_mean_buffer[client_id] = content['means']
        self.sample_count_buffer[client_id] = content['sample_counts']

        logger.info(f"Server: 收到 Client #{client_id} 的统计量")

        # 检查是否所有客户端都已上传
        if len(self.local_cov_buffer) == self.client_num:
            self._aggregate_and_broadcast()

    def _aggregate_and_broadcast(self):
        """聚合协方差矩阵并下发几何形状"""
        logger.info("Server: 开始聚合协方差矩阵...")

        # 获取所有类别
        all_classes = set()
        for client_stats in self.local_cov_buffer.values():
            all_classes.update(client_stats.keys())

        # 对每个类别聚合
        for class_id in all_classes:
            # 收集该类别的所有局部统计量
            local_covs = []
            local_means = []
            sample_counts = []

            for client_id in self.local_cov_buffer.keys():
                if class_id in self.local_cov_buffer[client_id]:
                    local_covs.append(self.local_cov_buffer[client_id][class_id])
                    local_means.append(self.local_mean_buffer[client_id][class_id])
                    sample_counts.append(self.sample_count_buffer[client_id][class_id])

            # 聚合计算全局协方差矩阵（公式4）
            global_cov, global_mean = aggregate_covariances(
                local_covs, local_means, sample_counts
            )
            self.global_cov_matrices[class_id] = global_cov

            # 特征值分解
            eigenvalues, eigenvectors = eigendecompose(global_cov)
            self.global_geometries[class_id] = {
                'eigenvalues': eigenvalues,
                'eigenvectors': eigenvectors
            }

            logger.info(f"Server: 类别 {class_id} 全局几何形状计算完成")

        # 下发全局几何形状
        self._broadcast_global_geometries()

        # 如果是多域场景，还需要下发域原型
        if self._cfg.ggeur.scenario == 'multi_domain':
            self._collect_and_broadcast_prototypes()

    def _broadcast_global_geometries(self):
        """下发全局几何形状到所有客户端"""
        logger.info("Server: 下发全局几何形状...")

        self.comm_manager.send(
            Message(
                msg_type='global_geometries',
                sender=self.ID,
                receiver=list(self.local_cov_buffer.keys()),
                state=self.state,
                content=self.global_geometries
            )
        )

    def _collect_and_broadcast_prototypes(self):
        """收集并下发域原型（多域场景）"""
        logger.info("Server: 收集域原型...")

        # 域原型 = 每个客户端的局部均值
        self.domain_prototypes = self.local_mean_buffer.copy()

        # 下发给所有客户端
        self.comm_manager.send(
            Message(
                msg_type='domain_prototypes',
                sender=self.ID,
                receiver=list(self.local_cov_buffer.keys()),
                state=self.state,
                content=self.domain_prototypes
            )
        )
```

---

### 阶段 5: 训练器实现（第 6 周）

#### Task 5.1: GGEURTrainer 实现
**文件**: `federatedscope/contrib/trainer/ggeur_trainer.py`

**功能**:
- 在增强后的嵌入数据上训练 MLP
- 支持标准的训练流程

**关键实现**:
```python
class GGEURTrainer(Trainer):
    """在嵌入空间训练 MLP 分类器"""

    def __init__(self, model, data, device, config, monitor):
        super().__init__(model, data, device, config, monitor)

        self.augmented_embeddings = None
        self.augmented_labels = None

        # 构建 MLP 分类器
        self.mlp_classifier = self._build_mlp()

    def _build_mlp(self):
        """构建 MLP 分类器"""
        input_dim = self._cfg.ggeur.embedding_dim  # CLIP 嵌入维度（如 512）
        hidden_dim = self._cfg.ggeur.mlp_hidden_dim
        num_classes = self._cfg.model.num_classes

        if self._cfg.ggeur.mlp_layers == 1:
            # 单层 MLP
            return nn.Linear(input_dim, num_classes)
        else:
            # 多层 MLP
            layers = []
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())

            for _ in range(self._cfg.ggeur.mlp_layers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                layers.append(nn.ReLU())

            layers.append(nn.Linear(hidden_dim, num_classes))
            return nn.Sequential(*layers)

    def setup_augmented_data(self, augmented_data):
        """设置增强后的数据"""
        self.augmented_embeddings = augmented_data['embeddings']
        self.augmented_labels = augmented_data['labels']

        # 创建 DataLoader
        from torch.utils.data import TensorDataset, DataLoader

        dataset = TensorDataset(self.augmented_embeddings,
                               self.augmented_labels)
        self.train_loader = DataLoader(
            dataset,
            batch_size=self._cfg.dataloader.batch_size,
            shuffle=True
        )

    def train(self):
        """训练流程"""
        self.mlp_classifier.train()
        self.mlp_classifier.to(self.device)

        optimizer = torch.optim.SGD(
            self.mlp_classifier.parameters(),
            lr=self._cfg.train.optimizer.lr,
            momentum=self._cfg.train.optimizer.momentum,
            weight_decay=self._cfg.train.optimizer.weight_decay
        )

        criterion = nn.CrossEntropyLoss()

        for epoch in range(self._cfg.train.local_update_steps):
            total_loss = 0.0
            total_correct = 0
            total_samples = 0

            for embeddings, labels in self.train_loader:
                embeddings = embeddings.to(self.device)
                labels = labels.to(self.device)

                # 前向传播
                logits = self.mlp_classifier(embeddings)
                loss = criterion(logits, labels)

                # 反向传播
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # 统计
                total_loss += loss.item() * len(labels)
                pred = logits.argmax(dim=1)
                total_correct += (pred == labels).sum().item()
                total_samples += len(labels)

            avg_loss = total_loss / total_samples
            accuracy = total_correct / total_samples

            logger.info(f"Epoch {epoch+1}/{self._cfg.train.local_update_steps}: "
                       f"Loss={avg_loss:.4f}, Acc={accuracy:.4f}")

        return total_samples, self.mlp_classifier.state_dict()

    def evaluate(self, target_data_split_name='test'):
        """评估流程"""
        self.mlp_classifier.eval()

        # 对测试数据提取 CLIP 嵌入
        test_data = self.data[target_data_split_name]
        clip_extractor = CLIPExtractor(device=self.device)
        test_embeddings = clip_extractor.extract_dataset_features(test_data)
        test_labels = torch.tensor(test_data.targets)

        with torch.no_grad():
            logits = self.mlp_classifier(test_embeddings.to(self.device))
            pred = logits.argmax(dim=1).cpu()
            accuracy = (pred == test_labels).float().mean().item()

        return {
            f'{target_data_split_name}_acc': accuracy,
            f'{target_data_split_name}_total': len(test_labels)
        }
```

---

### 阶段 6: 配置系统（第 7 周）

#### Task 6.1: GGEUR 配置
**文件**: `federatedscope/core/configs/cfg_ggeur.py`

```python
from yacs.config import CfgNode as CN

def add_ggeur_config(cfg):
    """添加 GGEUR_Clip 相关配置"""

    # GGEUR_Clip 配置组
    cfg.ggeur = CN()

    # 基础设置
    cfg.ggeur.use = False  # 是否启用 GGEUR_Clip
    cfg.ggeur.scenario = 'single_domain'  # 'single_domain' 或 'multi_domain'

    # CLIP 设置
    cfg.ggeur.clip_model = 'ViT-B/16'  # CLIP 模型
    cfg.ggeur.embedding_dim = 512  # 嵌入维度
    cfg.ggeur.cache_embeddings = True  # 是否缓存嵌入

    # 数据增强参数
    cfg.ggeur.n_samples_per_original = 10  # 单域：每个原始样本生成数（N）
    cfg.ggeur.m_samples_per_prototype = 500  # 多域：每个原型生成数（M）
    cfg.ggeur.total_samples_per_class = 2000  # 单域：每个类别总样本数
    cfg.ggeur.step1_samples_per_class = 500  # 多域 Step1：每个类别总样本数

    # 几何形状设置
    cfg.ggeur.top_k_eigenvalues = None  # 保留前 k 个特征值（None=全部）

    # MLP 分类器设置
    cfg.ggeur.mlp_layers = 1  # MLP 层数
    cfg.ggeur.mlp_hidden_dim = 512  # 隐藏层维度

    # 通信设置
    cfg.ggeur.upload_statistics_round = 0  # 在第几轮上传统计量
    cfg.ggeur.start_augmentation_round = 1  # 在第几轮开始增强
```

#### Task 6.2: 示例配置文件

**文件**: `scripts/example_configs/ggeur_cifar10_label_skew.yaml`

```yaml
use_gpu: True
device: 0
early_stop:
  patience: 50

federate:
  mode: 'standalone'
  total_round_num: 100
  client_num: 10
  sample_client_num: 10
  make_global_eval: True
  method: 'ggeur'  # 使用 GGEUR_Clip 方法

data:
  root: 'data/'
  type: 'CIFAR-10'
  splits: [0.8, 0.1, 0.1]
  subsample: 1.0
  splitter: 'lda'  # Dirichlet 分布
  splitter_args: [{'alpha': 0.1}]  # β=0.1，标签偏斜

dataloader:
  batch_size: 64
  num_workers: 0

model:
  type: 'mlp'
  num_classes: 10

train:
  optimizer:
    lr: 0.01
    momentum: 0.9
    weight_decay: 0.00001
  local_update_steps: 10

ggeur:
  use: True
  scenario: 'single_domain'
  clip_model: 'ViT-B/16'
  embedding_dim: 512
  n_samples_per_original: 10
  total_samples_per_class: 2000
  mlp_layers: 1
  upload_statistics_round: 0
  start_augmentation_round: 1

eval:
  freq: 10
  metrics: ['acc']
```

**文件**: `scripts/example_configs/ggeur_digits_domain_skew.yaml`

```yaml
use_gpu: True
device: 0

federate:
  mode: 'standalone'
  total_round_num: 50
  client_num: 4  # 4个域
  sample_client_num: 4
  method: 'ggeur'

data:
  root: 'data/'
  type: 'Digits'  # MNIST, USPS, SVHN, SYN
  subsample: 0.1  # 每个域使用10%数据

model:
  type: 'mlp'
  num_classes: 10

ggeur:
  use: True
  scenario: 'multi_domain'
  clip_model: 'ViT-B/16'
  n_samples_per_original: 10
  m_samples_per_prototype: 500
  step1_samples_per_class: 500
  mlp_layers: 1

eval:
  freq: 5
  metrics: ['acc', 'std']
```

---

### 阶段 7: 集成与测试（第 8-9 周）

#### Task 7.1: Worker 注册
**文件**: 修改 `federatedscope/contrib/worker/__init__.py`

```python
from federatedscope.contrib.worker.ggeur_client import GGEURClient
from federatedscope.contrib.worker.ggeur_server import GGEURServer
from federatedscope.register import register_worker

def call_ggeur_worker(method):
    if method.lower() == 'ggeur':
        return {
            'client': GGEURClient,
            'server': GGEURServer,
        }
    return None

register_worker('ggeur', call_ggeur_worker)
```

#### Task 7.2: Trainer 注册
**文件**: 修改 `federatedscope/contrib/trainer/__init__.py`

```python
from federatedscope.contrib.trainer.ggeur_trainer import GGEURTrainer
from federatedscope.register import register_trainer

def call_ggeur_trainer(method):
    if method.lower() == 'ggeur':
        return GGEURTrainer
    return None

register_trainer('ggeur', call_ggeur_trainer)
```

#### Task 7.3: 单元测试

创建测试文件：`tests/test_ggeur.py`

```python
import unittest
import torch
from federatedscope.contrib.utils.geometry_utils import (
    compute_covariance_matrix,
    eigendecompose
)
from federatedscope.contrib.utils.covariance_aggregator import (
    aggregate_covariances
)
from federatedscope.contrib.data_augmentation.ggeur_augmentation import (
    GGEURAugmentation
)

class TestGGEUR(unittest.TestCase):

    def test_covariance_computation(self):
        """测试协方差矩阵计算"""
        embeddings = torch.randn(100, 512)
        cov = compute_covariance_matrix(embeddings)
        self.assertEqual(cov.shape, (512, 512))
        self.assertTrue(torch.allclose(cov, cov.T))  # 对称性

    def test_eigendecomposition(self):
        """测试特征值分解"""
        cov = torch.randn(512, 512)
        cov = cov @ cov.T  # 确保半正定
        eigenvalues, eigenvectors = eigendecompose(cov)
        self.assertEqual(len(eigenvalues), 512)
        self.assertEqual(eigenvectors.shape, (512, 512))

    def test_covariance_aggregation(self):
        """测试协方差聚合"""
        # 模拟3个客户端的数据
        local_covs = [torch.randn(512, 512) for _ in range(3)]
        local_means = [torch.randn(512) for _ in range(3)]
        sample_counts = [100, 150, 200]

        global_cov, global_mean = aggregate_covariances(
            local_covs, local_means, sample_counts
        )

        self.assertEqual(global_cov.shape, (512, 512))
        self.assertEqual(global_mean.shape, (512,))

    def test_ggeur_augmentation_single_domain(self):
        """测试单域 GGEUR_Clip 增强"""
        # 模拟全局几何形状
        global_geometries = {
            0: {
                'eigenvalues': torch.randn(512).abs(),
                'eigenvectors': torch.randn(512, 512)
            }
        }

        augmenter = GGEURAugmentation(global_geometries, 'single_domain')

        # 模拟局部嵌入
        embeddings = torch.randn(10, 512)
        labels = torch.zeros(10, dtype=torch.long)

        aug_emb, aug_lab = augmenter.augment_single_domain(
            embeddings, labels, class_id=0, N=5
        )

        self.assertEqual(aug_emb.shape, (50, 512))  # 10 * 5
        self.assertEqual(aug_lab.shape, (50,))

if __name__ == '__main__':
    unittest.main()
```

#### Task 7.4: 端到端测试

创建测试脚本：`scripts/test_ggeur_cifar10.sh`

```bash
#!/bin/bash

# 测试 GGEUR_Clip 在 CIFAR-10 上的运行

python federatedscope/main.py \
    --cfg scripts/example_configs/ggeur_cifar10_label_skew.yaml \
    seed 123 \
    outdir exp/ggeur_cifar10_test
```

---

### 阶段 8: 实验复现（第 10-12 周）

#### Task 8.1: 标签偏斜实验（CIFAR-10, CIFAR-100, Tiny-ImageNet）

**实验配置**:
- CIFAR-10: β ∈ {0.01, 0.03, 0.05, 0.07, 0.09, 0.1, 0.3, 0.5}
- CIFAR-100: β ∈ {0.01, 0.03, 0.05, 0.07, 0.09, 0.1, 0.3, 0.5}
- Tiny-ImageNet: β ∈ {0.1, 0.3, 0.5}

**对比方法**:
- FedAvg (CLIP+MLP)
- FedMix (CLIP+MLP)
- FEDGEN (CLIP+MLP)
- FedFA (CLIP+MLP)
- FedTPG
- FedCLIP

**预期结果**:
复现 Table 1, 2, 3 的结果

#### Task 8.2: 域偏斜实验（Digits, Office-Caltech, PACS）

**实验配置**:
- Digits: 4个客户端，每个对应一个域
- Office-Caltech: 4个客户端
- PACS: 4个客户端

**对比方法**:
- FedAvg (CLIP+MLP) + GGEUR
- SCAFFOLD (CLIP+MLP) + GGEUR
- MOON (CLIP+MLP) + GGEUR
- FedDyn (CLIP+MLP) + GGEUR
- FedOPT (CLIP+MLP) + GGEUR
- FedProto (CLIP+MLP) + GGEUR
- FedNTD (CLIP+MLP) + GGEUR

**预期结果**:
复现 Table 4, 5, 6 的结果

**消融实验**:
- GGEUR Step 1 only
- GGEUR Step 1 & 2

#### Task 8.3: 混合偏斜实验（Office-Home-LDS）

**实验配置**:
- β = 0.1
- 4个客户端，每个对应一个域

**预期结果**:
复现 Table 7 的结果

#### Task 8.4: 收敛性分析

**实验**:
绘制训练曲线，复现 Figure 8

#### Task 8.5: 大规模客户端实验

**实验配置**:
- CIFAR-10, β=0.1
- K ∈ {100, 300, 500}

**预期结果**:
复现 Table 10 的结果

---

## 四、关键技术难点与解决方案

### 难点 1: CLIP 特征提取效率
**问题**: 每轮都提取特征会很慢

**解决方案**:
- 在第一轮预先提取所有数据的嵌入并缓存
- 使用 `torch.no_grad()` 减少内存占用
- 支持批量处理

### 难点 2: 协方差矩阵计算的数值稳定性
**问题**: 高维数据的协方差矩阵可能不稳定

**解决方案**:
- 添加正则化项：`Σ + λI`
- 使用 SVD 代替直接特征值分解
- 归一化嵌入特征

### 难点 3: 内存占用
**问题**: 存储大量嵌入和增强数据

**解决方案**:
- 使用 float16 存储嵌入
- 按需生成增强数据（不全部存储）
- 清理不再需要的中间结果

### 难点 4: 多域场景的原型管理
**问题**: 需要跨客户端共享原型

**解决方案**:
- 服务器端统一管理原型
- 只发送必要的原型给客户端
- 原型也需要隐私保护（只发送均值，不发送原始数据）

---

## 五、验证标准

### 5.1 正确性验证
- [ ] 单域场景 CIFAR-10 (β=0.1) 准确率 ≥ 95%
- [ ] 单域场景 CIFAR-100 (β=0.1) 准确率 ≥ 77%
- [ ] 多域场景 Digits 平均准确率 ≥ 82%

### 5.2 性能验证
- [ ] 与论文 Table 1-7 的结果误差 < 1%
- [ ] 训练时间开销 < 10s/round（Digits数据集）

### 5.3 可扩展性验证
- [ ] 支持大规模客户端（K=500）
- [ ] 支持不同数据集
- [ ] 易于与其他方法集成

---

## 六、时间规划（12 周）

| 周次 | 任务 | 交付物 |
|------|------|--------|
| 1-2  | 阶段1：基础设施搭建 | CLIP提取器、几何工具、协方差聚合器 |
| 3    | 阶段2：GGEUR算法 | 单域和多域增强算法 |
| 4    | 阶段3：客户端实现 | GGEURClient完整实现 |
| 5    | 阶段4：服务器实现 | GGEURServer完整实现 |
| 6    | 阶段5：训练器实现 | GGEURTrainer实现 |
| 7    | 阶段6：配置系统 | 配置文件和示例 |
| 8-9  | 阶段7：集成与测试 | 单元测试、端到端测试 |
| 10   | 阶段8.1：标签偏斜实验 | CIFAR-10/100, Tiny-ImageNet结果 |
| 11   | 阶段8.2-8.3：域偏斜和混合实验 | Digits, Office-Caltech, PACS, Office-Home-LDS结果 |
| 12   | 阶段8.4-8.5：补充实验 | 收敛性、大规模客户端实验 |

---

## 七、参考资料

1. **论文**: D:\Projects\FederatedScope\2503.06457v2_2025-12-03-17_03_33\2503.06457v2.md
2. **源代码**: https://github.com/WeiDai-David/2025CVPR_GGEUR
3. **FederatedScope 文档**: https://federatedscope.io/
4. **CLIP 模型**: https://github.com/openai/CLIP

---

## 八、后续扩展方向

1. **优化数据增强策略**
   - 自适应选择 N 和 M 参数
   - 探索其他分布假设（如 t-分布）

2. **隐私保护增强**
   - 差分隐私保护协方差矩阵
   - 安全多方计算聚合

3. **与其他方法结合**
   - GGEUR + FedProx
   - GGEUR + SCAFFOLD

4. **应用到其他任务**
   - 目标检测
   - 语义分割
   - NLP 任务

---

## 九、常见问题与解答

**Q1: 为什么不直接使用原始图像进行增强？**
A: 论文指出，在原始图像空间计算协方差矩阵会遇到高维稀疏问题。使用 CLIP 嵌入可以得到更稳定的几何形状。

**Q2: 单域和多域场景的区别是什么？**
A: 单域场景只有标签偏斜，客户端数据来自同一分布但类别不平衡；多域场景同时存在标签偏斜和域偏斜，客户端数据来自不同域。

**Q3: 如何选择超参数 N 和 M？**
A: 论文中，单域场景每类总样本数为2000，多域场景 Step1 每类500，Step2 每个原型500。可以根据数据集大小调整。

**Q4: GGEUR 与 FedFA 的主要区别？**
A: FedFA 假设数据服从高斯分布，GGEUR 使用协方差矩阵的几何形状（特征值和特征向量），更准确地描述分布。

---

## 十、检查清单

开始实现前，请确保：
- [ ] 已安装 CLIP 库：`pip install git+https://github.com/openai/CLIP.git`
- [ ] 已准备数据集：CIFAR-10, CIFAR-100, Tiny-ImageNet, Digits, Office-Caltech, PACS, Office-Home
- [ ] 已熟悉 FederatedScope 框架基本用法
- [ ] 已阅读论文并理解核心算法
- [ ] GPU 环境可用（建议 ≥ 16GB 显存）

---

**创建日期**: 2025-12-11
**最后更新**: 2025-12-11
**状态**: 待实施
