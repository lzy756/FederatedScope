# GGEUR过拟合问题分析和解决方案

## 问题诊断

### 症状
- 训练准确率：10% → 44%（持续提升）
- 测试准确率：约2%（几乎不变，接近随机）
- 增强数据量：125000-180000个样本

### 根本原因

**GGEUR生成了过多的合成数据**，导致模型过拟合到这些合成样本。

原始数据：
- 客户端1: 380样本 → 增强后125000样本（**扩大329倍**！）
- 客户端2: 982样本 → 增强后155000样本（**扩大158倍**！）
- 客户端3: 609样本 → 增强后140000样本（**扩大230倍**！）
- 客户端4: 1157样本 → 增强后180000样本（**扩大156倍**！）

## 与论文源码对比

### 论文源码的增强策略

查看论文源码`FedAvg_GGEUR.py`（第81-82行）：

```python
original_features_path = os.path.join(base_dir, dataset_name,
    f'client_{client_idx}_class_{class_idx}/final_embeddings_filled.npy')
```

关键：论文使用**每个类别**的增强数据，而不是批量增强！

**论文的数据流**：
```
1. 对每个类别单独增强
2. 每个类别增强到固定数量（如50个样本）
3. 然后合并所有类别的数据用于训练
```

**当前实现的问题**：
```
1. 对所有原始样本都进行增强
2. 每个原始样本生成50个合成样本（n_samples_per_original=50）
3. 然后再对每个原型生成50个样本（m_samples_per_prototype=50）
4. 结果：数据量爆炸！
```

## 解决方案

### 方案1：调整增强参数（推荐）

修改配置文件中的GGEUR参数：

```yaml
ggeur:
  # 关键参数调整
  n_samples_per_original: 5  # 降低！从50 → 5
  m_samples_per_prototype: 5  # 降低！从50 → 5
  target_size_per_class: 50  # 保持不变，这是每个类的最终目标数量

  # 或者使用目标分布策略
  target_distribution: uniform  # 每个类固定数量
```

**预期效果**：
- 增强数据量：125000 → 约3250（减少约38倍）
- 避免过拟合到合成数据
- 保持合理的类别平衡

### 方案2：使用原论文的参数

根据论文Figure 4的实验设置，应该是：

```yaml
ggeur:
  n_samples_per_original: 0  # 不对原始样本增强？
  m_samples_per_prototype: 0  # 或使用非常小的值
  target_size_per_class: 50  # 每个类最终50个样本
  target_distribution: uniform  # 均匀分布
```

### 方案3：检查实现是否与论文一致

需要确认：
1. 论文是否真的使用`n_samples_per_original=50`?
2. 论文的增强是否分类别进行？
3. 论文的最终训练数据量是多少？

## 立即行动

### 步骤1：修改配置

```yaml
# scripts/example_configs/ggeur_officehome_lds.yaml
ggeur:
  n_samples_per_original: 5  # 改为5
  m_samples_per_prototype: 5  # 改为5
  target_size_per_class: 50
```

### 步骤2：重新训练

```bash
python run_ggeur.py --cfg scripts/example_configs/ggeur_officehome_lds.yaml
```

### 步骤3：观察结果

关注：
1. 增强后的数据量（应该大幅减少）
2. 训练/测试准确率的差距（应该缩小）
3. 测试准确率的绝对值（应该提升）

## 预期改进

**修改前**：
- 训练准确率：44%
- 测试准确率：2%
- 差距：42%（严重过拟合）

**修改后**：
- 训练准确率：30-40%
- 测试准确率：20-30%
- 差距：<10%（合理）

## 其他可能需要调整的参数

如果结果仍不理想，可以尝试：

```yaml
train:
  local_update_steps: 50  # 减少训练轮数（当前100）
  optimizer:
    lr: 0.001  # 降低学习率（当前0.01）
    weight_decay: 0.001  # 添加正则化

ggeur:
  mlp_dropout: 0.3  # 添加dropout
```

## 关键特征值问题

日志中的特征值异常小：
```
Top 5 eigenvalues: [9.999975e-07, ...]
```

这可能是因为`cov_regularization: 1e-06`太大。可以尝试：

```yaml
ggeur:
  cov_regularization: 1e-08  # 降低正则化
```

但首先应该解决数据量过大的问题！
