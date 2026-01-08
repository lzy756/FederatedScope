# GGEUR: Gaussian Geometry-guided Feature Expansion with Unified Representation

## 概述

GGEUR 是一种针对**跨域联邦学习 (Cross-Domain Federated Learning)** 设计的方法，旨在解决以下核心问题：

- **数据异构性 (Data Heterogeneity)**：不同客户端的数据来自不同的域（如照片、艺术画、卡通、素描）
- **隐私保护**：客户端不能直接共享原始数据
- **域偏移 (Domain Shift)**：训练数据和测试数据可能来自不同的域

## 核心思想

GGEUR 的核心思想是：**利用预训练的视觉-语言模型（如CLIP）提取统一的特征表示，然后通过高斯几何建模进行特征扩展，从而在不共享原始数据的情况下实现跨域知识迁移。**

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           GGEUR 整体流程                                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Round 0: 统计量收集                                                         │
│  ┌─────────┐   ┌─────────┐   ┌─────────┐   ┌─────────┐                     │
│  │Client 1 │   │Client 2 │   │Client 3 │   │Client 4 │                     │
│  │(Photo)  │   │(Art)    │   │(Cartoon)│   │(Sketch) │                     │
│  └────┬────┘   └────┬────┘   └────┬────┘   └────┬────┘                     │
│       │             │             │             │                           │
│       ▼             ▼             ▼             ▼                           │
│  ┌─────────────────────────────────────────────────────┐                   │
│  │ 1. CLIP 特征提取  →  2. 计算局部统计量 (μ, Σ, n)    │                   │
│  └─────────────────────────────────────────────────────┘                   │
│       │             │             │             │                           │
│       └─────────────┴──────┬──────┴─────────────┘                           │
│                            ▼                                                │
│                    ┌──────────────┐                                         │
│                    │    Server    │                                         │
│                    │  聚合协方差   │                                         │
│                    └──────┬───────┘                                         │
│                           │                                                 │
│       ┌───────────────────┼───────────────────┐                             │
│       ▼                   ▼                   ▼                             │
│  Round 1-N: 训练                                                            │
│  ┌─────────┐        ┌─────────┐        ┌─────────┐                         │
│  │ Client  │        │ Client  │        │ Client  │                         │
│  │ 高斯增强 │        │ 高斯增强 │        │ 高斯增强 │                         │
│  │ MLP训练  │        │ MLP训练  │        │ MLP训练  │                         │
│  └────┬────┘        └────┬────┘        └────┬────┘                         │
│       │                  │                  │                               │
│       └──────────────────┼──────────────────┘                               │
│                          ▼                                                  │
│                   ┌──────────────┐                                          │
│                   │   FedAvg     │                                          │
│                   │  模型聚合    │                                          │
│                   └──────────────┘                                          │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 详细步骤

### 第一阶段：特征提取与统计量计算 (Round 0)

#### 1.1 CLIP 特征提取

每个客户端使用预训练的 CLIP 模型（如 ViT-B/16）从本地图像中提取特征：

```python
# 对于客户端 k 的每张图像 x_i
feature_i = CLIP.encode_image(x_i)  # 输出: 512维向量
```

**为什么使用 CLIP？**
- CLIP 在大规模图像-文本对上预训练，学到了**域不变的语义特征**
- 不同域的同类图像在 CLIP 特征空间中会靠近
- 提供了一个**统一的特征表示空间**

#### 1.2 计算局部统计量

对于每个类别 $c$，客户端 $k$ 计算：

**类别均值（原型 Prototype）：**
$$\mu_c^{(k)} = \frac{1}{n_c^{(k)}} \sum_{i: y_i = c} f_i$$

**类别协方差矩阵：**
$$\Sigma_c^{(k)} = \frac{1}{n_c^{(k)}} \sum_{i: y_i = c} (f_i - \mu_c^{(k)})(f_i - \mu_c^{(k)})^T$$

**样本数量：**
$$n_c^{(k)}$$

```python
# 代码实现
for class_idx, features in local_features.items():
    n = features.shape[0]
    mean = np.mean(features, axis=0)           # μ: (512,)
    centered = features - mean
    cov = (1.0 / n) * np.dot(centered.T, centered)  # Σ: (512, 512)

    local_means[class_idx] = mean
    local_covs[class_idx] = cov
    local_counts[class_idx] = n
```

#### 1.3 上传统计量到服务器

每个客户端上传：
- 类别均值 $\mu_c^{(k)}$（也作为原型使用）
- 协方差矩阵 $\Sigma_c^{(k)}$
- 样本数量 $n_c^{(k)}$

**隐私保护**：只上传统计量，不上传原始数据或特征！

---

### 第二阶段：服务器聚合 (Server Aggregation)

#### 2.1 使用平行轴定理聚合协方差矩阵

服务器收到所有客户端的统计量后，使用**平行轴定理 (Parallel Axis Theorem)** 聚合协方差矩阵：

```
┌─────────────────────────────────────────────────────────────────┐
│                     平行轴定理 (Parallel Axis Theorem)           │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  全局协方差 = 局部协方差的加权平均 + 均值差异产生的方差            │
│                                                                 │
│  Σ_c^{global} = (1/N) * Σ_{k} n_c^{(k)} * Σ_c^{(k)}             │
│               + (1/N) * Σ_{k} n_c^{(k)} * (μ_c^{(k)} - μ_c^{global})(...)^T │
│                                                                 │
│  其中 N = Σ_{k} n_c^{(k)} 是该类别的总样本数                     │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

**数学公式：**

1. 计算全局均值：
$$\mu_c^{global} = \frac{1}{N_c} \sum_k n_c^{(k)} \mu_c^{(k)}$$

2. 计算全局协方差：
$$\Sigma_c^{global} = \underbrace{\frac{1}{N_c} \sum_k n_c^{(k)} \Sigma_c^{(k)}}_{\text{局部协方差的加权平均}} + \underbrace{\frac{1}{N_c} \sum_k n_c^{(k)} (\mu_c^{(k)} - \mu_c^{global})(\mu_c^{(k)} - \mu_c^{global})^T}_{\text{客户端间的均值差异}}$$

```python
# 代码实现
def _aggregate_covariances(self):
    for class_idx in all_classes:
        # 收集各客户端的统计量
        means, covs, counts = [], [], []
        for client_stats in local_statistics_buffer.values():
            if class_idx in client_stats['means']:
                means.append(client_stats['means'][class_idx])
                covs.append(client_stats['covs'][class_idx])
                counts.append(client_stats['counts'][class_idx])

        total_count = sum(counts)

        # 计算全局均值
        aggregated_mean = sum(c * m for c, m in zip(counts, means)) / total_count

        # 第一项：局部协方差的加权平均
        aggregated_cov = sum(c * cov for c, cov in zip(counts, covs))

        # 第二项：均值差异产生的方差
        for mean, count in zip(means, counts):
            diff = mean - aggregated_mean
            aggregated_cov += count * np.outer(diff, diff)

        aggregated_cov /= total_count
        global_cov_matrices[class_idx] = aggregated_cov
```

#### 2.2 准备跨客户端原型

服务器为每个客户端准备**其他客户端的原型**，用于后续的跨域特征增强：

```python
# 对于客户端 k，收集其他客户端的原型
other_prototypes[k] = {
    class_idx: [prototypes from other clients]
    for class_idx in all_classes
}
```

#### 2.3 广播全局协方差和原型

服务器向所有客户端广播：
- 全局协方差矩阵 $\Sigma_c^{global}$
- 其他客户端的原型 $\{\mu_c^{(j)}\}_{j \neq k}$

---

### 第三阶段：高斯特征增强 (Gaussian Feature Augmentation)

这是 GGEUR 的核心创新。每个客户端使用全局协方差矩阵生成增强特征。

#### 3.1 基本原理

假设每个类别的特征服从多元高斯分布：
$$f \sim \mathcal{N}(\mu_c, \Sigma_c^{global})$$

我们可以从这个分布中采样生成新的特征：

```
┌─────────────────────────────────────────────────────────────────┐
│                     高斯特征增强示意图                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│    原始特征空间                      增强后的特征空间              │
│                                                                 │
│        ●                                 ● ○ ○                  │
│      ● ●                               ● ● ○ ○ ○                │
│    ●   ●  ●     ────────────────>    ● ● ● ○ ○ ○ ○              │
│      ●                               ● ● ○ ○ ○                  │
│        ●                                 ● ○ ○                  │
│                                                                 │
│   ● 原始特征                          ● 原始特征                  │
│                                       ○ 生成的特征               │
│                                                                 │
│   使用全局协方差 Σ_global 在原始特征周围生成更多样本              │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

#### 3.2 特征生成方法

**方法一：围绕本地特征生成**

以每个原始特征为中心，使用全局协方差生成新特征：

$$f_{new} \sim \mathcal{N}(f_{original}, \Sigma_c^{global})$$

```python
for feat in local_features[class_idx]:
    generated = generate_samples(
        mean=feat,                    # 以原始特征为中心
        cov=global_cov_matrices[class_idx],  # 使用全局协方差
        num_samples=num_generated_per_sample
    )
```

**方法二：围绕其他客户端的原型生成**

使用其他客户端的类别原型（均值）作为中心，生成该类别的特征：

$$f_{new} \sim \mathcal{N}(\mu_c^{(j)}, \Sigma_c^{global}), \quad j \neq k$$

```python
if class_idx in other_prototypes:
    for prototype in other_prototypes[class_idx]:
        generated = generate_samples(
            mean=prototype,           # 以其他客户端的原型为中心
            cov=global_cov_matrices[class_idx],
            num_samples=num_generated_per_prototype
        )
```

#### 3.3 高斯采样的数学实现

为确保协方差矩阵正定，使用 Cholesky 分解：

$$\Sigma = LL^T$$

然后生成样本：
$$f_{new} = \mu + L \cdot z, \quad z \sim \mathcal{N}(0, I)$$

```python
def _generate_samples(self, mean, cov_matrix, num_samples):
    # 确保正定性
    cov_matrix = nearest_positive_definite(cov_matrix)

    # 添加数值稳定性
    jitter = 1e-6
    while True:
        try:
            L = np.linalg.cholesky(cov_matrix + jitter * np.eye(dim))
            break
        except np.linalg.LinAlgError:
            jitter *= 10

    # 采样
    samples = np.random.multivariate_normal(mean, L @ L.T, num_samples)
    return samples
```

#### 3.4 增强数据的组成

最终的增强数据集包含：

```
┌─────────────────────────────────────────────────────────────────┐
│                    增强数据集组成                                │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1. 原始特征（保留所有本地数据）                                  │
│     └── local_features[class]                                   │
│                                                                 │
│  2. 围绕原始特征生成的特征（扩展本地数据分布）                     │
│     └── 每个原始特征 → num_generated_per_sample 个新特征         │
│                                                                 │
│  3. 围绕其他客户端原型生成的特征（引入跨域知识）                   │
│     └── 每个原型 → num_generated_per_prototype 个新特征          │
│                                                                 │
│  最终采样到 target_size_per_class 个样本（如果超过）              │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

### 第四阶段：MLP分类器训练 (FedAvg)

#### 4.1 MLP架构

在增强后的CLIP特征上训练一个简单的MLP分类器：

```
输入: CLIP特征 (512维)
      ↓
[可选] 隐藏层 (Linear → ReLU → Dropout)
      ↓
输出层 (Linear → num_classes)
      ↓
输出: 类别logits
```

```python
if hidden_dim > 0:
    mlp_classifier = nn.Sequential(
        nn.Linear(input_dim, hidden_dim),  # 512 → hidden_dim
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, num_classes)  # hidden_dim → num_classes
    )
else:
    # 简单线性分类器
    mlp_classifier = nn.Linear(input_dim, num_classes)  # 512 → num_classes
```

#### 4.2 本地训练

每个客户端在增强数据集上训练MLP：

```python
for epoch in range(local_epochs):
    for features, labels in augmented_loader:
        outputs = mlp_classifier(features)
        loss = CrossEntropyLoss(outputs, labels)
        loss.backward()
        optimizer.step()
```

#### 4.3 FedAvg聚合

服务器收集所有客户端的MLP参数，使用FedAvg聚合：

$$\theta^{global} = \sum_k \frac{n_k}{N} \theta^{(k)}$$

其中 $n_k$ 是客户端 $k$ 的样本数，$N$ 是总样本数。

---

## 关键参数

| 参数 | 含义 | 推荐值 |
|------|------|--------|
| `num_generated_per_sample` | 每个原始样本生成的新样本数 | 50 |
| `num_generated_per_prototype` | 每个原型生成的样本数 | 50 |
| `target_size_per_class` | 每类的目标样本数 | 50-100 |
| `mlp_hidden_dim` | MLP隐藏层维度（0=线性分类器） | 0 或 256 |
| `embedding_dim` | CLIP特征维度 | 512 |

---

## 训练流程总结

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           GGEUR 完整训练流程                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Round 0:                                                                   │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  Clients:                                                           │   │
│  │    1. 加载CLIP模型                                                   │   │
│  │    2. 提取本地图像的CLIP特征                                          │   │
│  │    3. 计算每类的均值μ、协方差Σ、样本数n                                │   │
│  │    4. 上传统计量到服务器                                              │   │
│  │                                                                     │   │
│  │  Server:                                                            │   │
│  │    5. 使用平行轴定理聚合协方差矩阵                                     │   │
│  │    6. 广播全局协方差和其他客户端原型                                   │   │
│  │                                                                     │   │
│  │  Clients:                                                           │   │
│  │    7. 接收全局协方差和原型                                            │   │
│  │    8. 执行高斯特征增强                                                │   │
│  │    9. 创建增强数据集                                                  │   │
│  │   10. 构建MLP分类器                                                   │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  Round 1 ~ N:                                                               │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  每轮:                                                               │   │
│  │    1. Server 广播全局MLP参数                                          │   │
│  │    2. Clients 在增强数据上本地训练MLP                                  │   │
│  │    3. Clients 上传更新后的MLP参数                                     │   │
│  │    4. Server 使用FedAvg聚合参数                                       │   │
│  │    5. Server 在测试集上评估模型                                       │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  最终输出: 训练好的MLP分类器（可与CLIP结合进行推理）                         │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 推理流程

```python
# 推理时
def predict(image):
    # 1. 使用CLIP提取特征
    feature = clip_model.encode_image(image)  # (512,)

    # 2. 使用训练好的MLP分类
    logits = mlp_classifier(feature)  # (num_classes,)

    # 3. 获取预测类别
    predicted_class = logits.argmax()

    return predicted_class
```

---

## 方法优势

1. **隐私保护**：只共享统计量，不共享原始数据
2. **跨域知识迁移**：通过共享协方差和原型实现
3. **数据增强**：在特征空间进行增强，更高效
4. **统一表示**：CLIP提供域不变的特征空间
5. **通信高效**：只需在Round 0交换统计量
6. **模型轻量**：训练简单的MLP分类器

---

## 与FedAvg的对比

| 方面 | FedAvg | GGEUR |
|------|--------|-------|
| 训练目标 | 完整CNN模型 | MLP分类器 |
| 特征提取 | 学习得到 | CLIP（冻结） |
| 跨域能力 | 有限 | 强（CLIP预训练） |
| 数据增强 | 无 | 高斯特征增强 |
| 知识共享 | 模型参数 | 统计量 + 参数 |
| 通信开销 | 每轮模型 | Round 0统计量 + 每轮MLP |

---

## 数学符号总结

| 符号 | 含义 |
|------|------|
| $f_i$ | 第i个样本的CLIP特征向量 |
| $y_i$ | 第i个样本的标签 |
| $\mu_c^{(k)}$ | 客户端k的类别c均值（原型） |
| $\Sigma_c^{(k)}$ | 客户端k的类别c协方差矩阵 |
| $n_c^{(k)}$ | 客户端k的类别c样本数 |
| $\Sigma_c^{global}$ | 全局聚合的类别c协方差矩阵 |
| $\mathcal{N}(\mu, \Sigma)$ | 均值为μ、协方差为Σ的多元高斯分布 |

---

## 参考实现

- 客户端实现: `federatedscope/contrib/worker/ggeur_client.py`
- 服务器实现: `federatedscope/contrib/worker/ggeur_server.py`
- 配置文件: `federatedscope/core/configs/cfg_ggeur.py`
- 示例配置: `scripts/example_configs/ggeur_*.yaml`
