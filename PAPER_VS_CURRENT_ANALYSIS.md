# 论文源码 vs 当前实现：GGEUR数据增强对比分析

## 论文源码的数据增强策略

### 关键代码分析

**文件**: `prototype_cov_matrix_generate_features.py`

#### 1. 数据增强函数 (第29-39行)

```python
def expand_features_with_cov(original_features, cov_matrix, num_per_sample):
    generated_samples = []
    for feature in original_features:
        new_samples = generate_new_samples(feature, cov_matrix, num_per_sample)
        generated_samples.append(new_samples)

    all_generated_samples = np.vstack(generated_samples)

    # 关键！从生成的样本中随机选择50个补全样本
    selected_indices = np.random.choice(all_generated_samples.shape[0], 50, replace=False)
    return all_generated_samples[selected_indices]
```

**重要**：虽然每个原始样本生成50个新样本，但**最后只选择50个**！

#### 2. 合并函数 (第42-69行)

```python
def combine_samples(original_features, expanded_features, other_generated_samples, target_size=50):
    num_original = original_features.shape[0]  # 原始样本数量

    # 如果原始样本数量大于目标数量，从原始样本中抽取 target_size 数量的样本
    if num_original >= target_size:
        selected_indices = np.random.choice(num_original, target_size, replace=False)
        return original_features[selected_indices]

    # 需要从生成样本中补充的数量
    num_needed = target_size - num_original

    # ... 用生成样本补充到50个
    final_samples = np.vstack((original_features, combined_generated_samples[selected_indices]))

    return final_samples
```

**核心逻辑**：**每个类最终固定为50个样本**！

#### 3. 处理流程 (第72-102行)

```python
def process_clients(client_id, class_idx, original_features, cov_matrices, prototype_features, num_per_sample):
    # 1. 从原始特征生成补全样本（选50个）
    if original_features.shape[0] > 0:
        expanded_features = expand_features_with_cov(original_features, cov_matrices[class_idx], num_per_sample)

    # 2. 从其他客户端的原型生成样本（每个原型生成50个）
    for prototype in prototype_features:
        new_samples = generate_new_samples(prototype, cov_matrices[class_idx], 50)
        additional_samples.append(new_samples)

    # 3. 合并到最终的50个样本
    final_samples = combine_samples(original_features, expanded_features, additional_samples, target_size=50)

    return final_samples
```

### 论文源码的最终数据量

**配置** (第114-119行):
```python
datasets_config = {
    'Art': {'client_ids': [0], 'num_generated_per_sample': 50},
    'Clipart': {'client_ids': [1], 'num_generated_per_sample': 50},
    'Product': {'client_ids': [2], 'num_generated_per_sample': 50},
    'Real_World': {'client_ids': [3], 'num_generated_per_sample': 50}
}
```

**数据量计算**：

根据日志，LDS后各客户端的非零样本类别数：
- 客户端1 (Art): 25个类 → 25 × 50 = **1,250个样本**
- 客户端2 (Clipart): 31个类 → 31 × 50 = **1,550个样本**
- 客户端3 (Product): 28个类 → 28 × 50 = **1,400个样本**
- 客户端4 (Real_World): 36个类 → 36 × 50 = **1,800个样本**

**关键点**：
1. 每个类固定50个样本
2. 只有非零样本的类才会被增强
3. 总数据量 = 非零样本类数 × 50

## 当前FederatedScope实现的问题

### 数据量对比

| 客户端 | 原始样本 | 非零类数 | **论文源码增强后** | **当前实现增强后** | 差异倍数 |
|--------|----------|----------|-------------------|-------------------|---------|
| 1 (Art) | 380 | 25 | **1,250** | 125,000 | **100倍** |
| 2 (Clipart) | 982 | 31 | **1,550** | 155,000 | **100倍** |
| 3 (Product) | 609 | 28 | **1,400** | 140,000 | **100倍** |
| 4 (Real_World) | 1,157 | 36 | **1,800** | 180,000 | **100倍** |

### 当前实现的错误逻辑

从日志推断，当前实现可能是：

```python
# 错误的实现
for original_sample in train_data:
    # 对每个原始样本生成 n_samples_per_original 个新样本
    generated = generate_samples(original_sample, n_samples_per_original=50)
    augmented_data.append(generated)  # 保留所有生成的样本！

for prototype in prototypes:
    # 对每个原型生成 m_samples_per_prototype 个新样本
    generated = generate_samples(prototype, m_samples_per_prototype=50)
    augmented_data.append(generated)  # 保留所有生成的样本！

# 结果：数据量爆炸！
# 380个原始样本 × 50 + 额外的原型样本 × 50 = 大量样本
```

**问题**：
1. 没有"每个类固定50个样本"的限制
2. 保留了所有生成的样本
3. 数据量是论文的100倍

## 为什么论文源码能得到好效果？

### 1. 合理的数据量

**论文**: 每个类50个样本
- 客户端1: 1,250个样本（合理）
- 训练准确率和测试准确率接近
- 没有严重过拟合

**当前实现**: 每个类约2000个样本
- 客户端1: 125,000个样本（过多）
- 训练准确率44%，测试准确率2%
- 严重过拟合到合成数据

### 2. 保留原始样本的信息

论文的`combine_samples`函数：
```python
if num_original >= target_size:
    # 如果原始样本够多，直接从原始样本中选
    return original_features[selected_indices]
else:
    # 否则，保留所有原始样本，用生成样本补充
    final_samples = np.vstack((original_features, generated[selected_indices]))
```

**优势**：
- 优先保留真实数据
- 生成数据只用于补充
- 避免模型过度依赖合成数据

### 3. 类别平衡

**论文**: 每个类固定50个样本
- 所有非零样本类都有相同数量的训练数据
- 类别平衡良好

**当前实现**: 根据原始样本数量按比例增强
- 原本样本多的类增强后更多
- 可能导致类别不平衡

## 修复方案

### 方案1：完全匹配论文实现（推荐）

修改配置文件：

```yaml
ggeur:
  # 关键：每个类固定的目标样本数
  target_size_per_class: 50  # 每个类最终50个样本

  # 中间步骤的参数（虽然生成很多，但最后会筛选）
  n_samples_per_original: 50  # 保持不变
  m_samples_per_prototype: 50  # 保持不变

  # 重要：使用均匀目标分布
  target_distribution: uniform  # 确保每个类固定数量
```

**关键修改点**：在数据增强的最后，对每个类：
1. 如果原始样本 >= 50：随机抽取50个
2. 如果原始样本 < 50：保留所有原始样本 + 生成样本补充到50个

### 方案2：临时快速修复

如果无法修改实现代码，临时降低生成数量：

```yaml
ggeur:
  n_samples_per_original: 1  # 降低到1
  m_samples_per_prototype: 1  # 降低到1
  target_size_per_class: 50
```

**预期数据量**：
- 客户端1: 380 × 1 + 补充 ≈ 1,250（接近论文）

### 方案3：修改实现代码

在GGEUR的数据增强代码中添加：

```python
def augment_per_class(embeddings, labels, target_size=50):
    """按类别增强，每个类固定target_size个样本"""
    unique_classes = torch.unique(labels)
    augmented_embeddings = []
    augmented_labels = []

    for class_id in unique_classes:
        class_mask = (labels == class_id)
        class_embeddings = embeddings[class_mask]

        # 如果样本数 >= target_size，随机抽取
        if len(class_embeddings) >= target_size:
            indices = torch.randperm(len(class_embeddings))[:target_size]
            final_embeddings = class_embeddings[indices]
        else:
            # 否则，生成样本补充到target_size
            num_needed = target_size - len(class_embeddings)
            generated = generate_samples(class_embeddings, num_needed)
            final_embeddings = torch.cat([class_embeddings, generated])

        augmented_embeddings.append(final_embeddings)
        augmented_labels.append(torch.full((target_size,), class_id))

    return torch.cat(augmented_embeddings), torch.cat(augmented_labels)
```

## 预期改进

### 修改前
- 训练数据: 125,000样本
- 训练准确率: 44%
- 测试准确率: 2%
- 过拟合程度: 42%

### 修改后（匹配论文）
- 训练数据: 1,250样本（减少100倍）
- 训练准确率: 30-35%
- 测试准确率: 25-30%
- 过拟合程度: <10%

## 总结

**论文源码能得到好效果的核心原因**：

1. ✅ **每个类固定50个样本** - 避免数据爆炸
2. ✅ **优先保留原始样本** - 真实数据为主
3. ✅ **类别平衡** - 所有类平等对待
4. ✅ **合理的数据量** - 不会过拟合到合成数据

**当前实现的问题**：

1. ❌ **数据量是论文的100倍** - 严重过拟合
2. ❌ **没有按类别限制** - 数据爆炸
3. ❌ **保留所有生成样本** - 引入大量噪声

**立即行动**：
- 在GGEUR实现中添加**每个类固定数量**的逻辑
- 或者临时降低`n_samples_per_original`和`m_samples_per_prototype`到1
