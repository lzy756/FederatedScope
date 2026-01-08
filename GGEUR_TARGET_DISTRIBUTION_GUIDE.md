# GGEUR 目标分布功能使用指南

## 功能概述

此功能允许你在配置文件中指定目标数据分布，GGEUR 算法会生成符合该分布的增强数据。

### 主要特性

1. **自定义分布**：可以为每个类别指定不同的样本比例
2. **自动归一化**：输入的比例会自动归一化（不需要和为1）
3. **默认均匀分布**：如果不指定，则使用均匀分布
4. **分布验证**：增强后会显示目标分布 vs 实际分布的对比

## 配置说明

### 1. 配置文件参数

在 YAML 配置文件的 `ggeur` 部分添加：

```yaml
ggeur:
  # 目标分布：每个类别的样本比例
  # 长度必须等于类别数
  # 如果为 null，使用均匀分布（默认）
  target_distribution: [0.1, 0.15, 0.1, 0.15, 0.2, 0.15, 0.15]

  # 增强后的总样本数
  # 如果为 null，自动计算（推荐）
  total_samples_after_augmentation: null
```

### 2. 参数说明

#### `target_distribution`
- **类型**：列表（浮点数）或 null
- **长度**：必须等于数据集的类别数
- **值**：每个类别的目标比例（会自动归一化）
- **默认**：null（均匀分布）

**示例：**

```yaml
# PACS 数据集（7个类别）
# 均匀分布
target_distribution: null

# 不均匀分布（会自动归一化）
target_distribution: [1, 2, 1, 2, 3, 2, 2]  # 等价于 [0.077, 0.154, 0.077, 0.154, 0.231, 0.154, 0.154]

# 直接指定比例
target_distribution: [0.05, 0.1, 0.15, 0.2, 0.2, 0.15, 0.15]  # 和必须为1（或会自动归一化）
```

#### `total_samples_after_augmentation`
- **类型**：整数或 null
- **说明**：增强后的总样本数
- **默认**：null（自动计算）
  - 单域场景：`total_samples_per_class * num_classes`
  - 多域场景：`step1_samples_per_class * num_classes`

**示例：**

```yaml
# 自动计算（推荐）
total_samples_after_augmentation: null

# 手动指定（例如：7个类，总共3500个样本）
total_samples_after_augmentation: 3500
```

## 使用示例

### 示例 1：均匀分布（默认）

```yaml
ggeur:
  target_distribution: null
  # 每个类别的样本数量相同
```

**输出：**
```
Using uniform distribution (default)
Class 0: 500 samples (14.29%)
Class 1: 500 samples (14.29%)
Class 2: 500 samples (14.29%)
...
```

### 示例 2：长尾分布

```yaml
ggeur:
  # 模拟长尾分布：少数类别有很多样本，多数类别样本较少
  target_distribution: [0.3, 0.2, 0.15, 0.1, 0.1, 0.08, 0.07]
  total_samples_after_augmentation: 3500
```

**输出：**
```
Target distribution specified: [0.3, 0.2, 0.15, 0.1, 0.1, 0.08, 0.07]
Target samples per class: [1050, 700, 525, 350, 350, 280, 245]

Class    Original     Augmented    Proportion   Ratio
------------------------------------------------------------------------
0        150          1050         0.3000       7.00x
1        150          700          0.2000       4.67x
2        150          525          0.1500       3.50x
3        150          350          0.1000       2.33x
4        150          350          0.1000       2.33x
5        150          280          0.0800       1.87x
6        150          245          0.0700       1.63x
```

### 示例 3：平衡不均衡数据

假设原始数据不均衡：

```python
# 原始分布
Class 0: 50 samples
Class 1: 200 samples
Class 2: 100 samples
Class 3: 300 samples
```

使用均匀目标分布来平衡：

```yaml
ggeur:
  target_distribution: null  # 均匀分布
  total_samples_after_augmentation: 2000
```

**结果：**
```
Class    Original     Augmented    Proportion   Ratio
------------------------------------------------------------------------
0        50           500          0.2500       10.00x
1        200          500          0.2500       2.50x
2        100          500          0.2500       5.00x
3        300          500          0.2500       1.67x
```

## 输出说明

### 数据分布报告

运行时会在每个客户端增强数据后输出详细报告：

```
================================================================================
CLIENT #1 - DATA DISTRIBUTION REPORT
================================================================================

Original data: 1050 samples
Augmented data: 3500 samples
Augmentation ratio: 3.33x

Per-class distribution:
Class    Original     Augmented    Proportion   Ratio
--------------------------------------------------------------------------------
0        150          1050         0.3000       7.00x
1        150          700          0.2000       4.67x
2        150          525          0.1500       3.50x
3        150          350          0.1000       2.33x
4        150          350          0.1000       2.33x
5        150          280          0.0800       1.87x
6        150          245          0.0700       1.63x

Target vs Actual distribution:
Class    Target       Actual       Difference
--------------------------------------------------------------------------------
0        0.3000       0.3000       +0.0000
1        0.2000       0.2000       +0.0000
2        0.1500       0.1500       +0.0000
3        0.1000       0.1000       +0.0000
4        0.1000       0.1000       +0.0000
5        0.0800       0.0800       +0.0000
6        0.0700       0.0700       +0.0000
================================================================================
```

### 评估结果输出

每轮评估后会显示格式化的准确率：

```
================================================================================
ROUND 5 - CLIENT #1 - EVALUATION RESULTS
================================================================================
TEST     Accuracy: 0.9245 (92.45%)
================================================================================
```

## 完整配置示例

### PACS 数据集 - 均匀分布

```yaml
# scripts/example_configs/ggeur_pacs_uniform.yaml

use_gpu: True
device: 0

federate:
  mode: 'standalone'
  total_round_num: 50
  client_num: 4
  sample_client_num: 4
  method: 'ggeur'

data:
  root: 'data/'
  type: 'PACS'
  splits: [0.3, 0.0, 0.0]
  splitter: 'domain'
  batch_size: 16

model:
  type: 'mlp'
  num_classes: 7

train:
  local_update_steps: 10
  batch_or_epoch: 'epoch'
  optimizer:
    type: 'SGD'
    lr: 0.001
    momentum: 0.9
    weight_decay: 0.00001

ggeur:
  use: True
  scenario: 'multi_domain'

  clip_model: 'ViT-B/16'
  embedding_dim: 512
  cache_embeddings: True

  n_samples_per_original: 12
  m_samples_per_prototype: 500
  step1_samples_per_class: 500

  use_full_covariance: True
  eigenvalue_scaling: True

  # 均匀分布（默认）
  target_distribution: null
  total_samples_after_augmentation: null

eval:
  freq: 5
  metrics: ['acc']
  best_res_update_round_wise_key: 'acc'

trainer:
  type: 'ggeur'

outdir: 'exp/ggeur_pacs_uniform'
expname: 'ggeur_pacs_uniform'
seed: 12345
```

### PACS 数据集 - 长尾分布

```yaml
# scripts/example_configs/ggeur_pacs_longtail.yaml

# ... (其他配置相同)

ggeur:
  use: True
  scenario: 'multi_domain'

  # ... (其他参数相同)

  # 长尾分布
  target_distribution: [0.3, 0.2, 0.15, 0.1, 0.1, 0.08, 0.07]
  total_samples_after_augmentation: 3500  # 7 classes * 500 samples

outdir: 'exp/ggeur_pacs_longtail'
expname: 'ggeur_pacs_longtail'
```

## 注意事项

### 1. 分布长度必须匹配类别数

```yaml
# ❌ 错误：PACS有7个类，但只提供了5个比例
target_distribution: [0.2, 0.2, 0.2, 0.2, 0.2]

# ✅ 正确：提供7个比例
target_distribution: [0.15, 0.15, 0.15, 0.15, 0.15, 0.125, 0.125]
```

### 2. 比例会自动归一化

```yaml
# 这两种写法等价：
target_distribution: [1, 2, 3, 4]
target_distribution: [0.1, 0.2, 0.3, 0.4]
```

### 3. 推荐使用自动计算总样本数

```yaml
# ✅ 推荐：让系统自动计算
total_samples_after_augmentation: null

# ⚠️ 手动指定：需要确保与增强参数匹配
total_samples_after_augmentation: 3500
```

### 4. 极端分布可能导致增强失败

```yaml
# ⚠️ 警告：极小的比例可能导致某些类别样本不足
target_distribution: [0.9, 0.02, 0.02, 0.02, 0.02, 0.01, 0.01]
```

如果某个类别的原始样本数已经大于目标样本数，会从原始样本中随机抽样。

## 调试和验证

### 1. 检查分布是否正确实现

查看输出的 "Target vs Actual distribution" 表格：
- `Difference` 应该接近 0.0000
- 如果差异较大（> 0.01），检查配置和数据

### 2. 监控增强比例

每个类别的 `Ratio` 列显示增强倍数：
- 均匀分布：所有类别的 Ratio 应该相近
- 不均匀分布：Ratio 会根据目标分布调整

### 3. 验证总样本数

```
Original data: 1050 samples
Augmented data: 3500 samples
```

确认增强后的总样本数符合预期。

## 常见问题

**Q: 如何实现完全均匀的分布？**

A: 设置 `target_distribution: null`（默认）

**Q: 可以指定每个类别的绝对样本数吗？**

A: 不能直接指定，但可以通过比例和总样本数实现：
```yaml
target_distribution: [0.3, 0.2, 0.5]  # 比例
total_samples_after_augmentation: 1000  # 总数
# 结果：300, 200, 500 样本
```

**Q: 如果原始数据已经很不平衡怎么办？**

A: GGEUR 会根据目标分布调整每个类别的增强程度，自动平衡数据。

**Q: 分布差异为什么不是完全0？**

A: 由于舍入误差和样本的离散性，可能有微小差异（通常 < 0.001），这是正常的。

## 技术细节

### 算法流程

1. 解析目标分布，归一化为概率
2. 根据总样本数计算每个类别的目标样本数
3. 对每个类别独立执行 GGEUR 增强，直到达到目标样本数
4. 验证实际分布与目标分布的差异
5. 输出详细的分布报告

### 实现位置

- 配置：`federatedscope/core/configs/cfg_ggeur.py`
- 增强逻辑：`federatedscope/contrib/data_augmentation/ggeur_augmentation.py`
- 客户端：`federatedscope/contrib/worker/ggeur_client.py`

## 扩展阅读

- [GGEUR 完整修复总结](GGEUR_FIX_SUMMARY.md)
- [GGEUR 实现差异分析](GGEUR_IMPLEMENTATION_DIFFERENCES.md)
- [GGEUR 配置示例](scripts/example_configs/ggeur_pacs_multi_domain.yaml)
