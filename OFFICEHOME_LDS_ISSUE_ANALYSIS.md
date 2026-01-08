# OfficeHome-LDS 数据分布问题分析

## 问题描述

使用 `ggeur_officehome_lds.yaml` 配置训练时，初始准确率很低（约5%），怀疑数据集分布与论文源码不一致。

## 关键发现

### 1. 论文源码的LDS实现方式

在论文源码 `ProjectsFederatedScopeGGEUR_source/Multi Domain/Office-Home-LDS/` 中：

#### 核心算法

```python
# 生成 4x65 的Dirichlet分布矩阵
def generate_dirichlet_matrix(alpha):
    return dirichlet([alpha] * 4, 65).T  # 4行（客户端） x 65列（类别）

# 对每个域/客户端，使用矩阵的一行来分配65个类的样本
def split_samples_for_domain(class_train_indices, dirichlet_column):
    client_indices = []
    for class_label, indices in class_train_indices.items():
        num_samples = len(indices)
        proportion = dirichlet_column[class_label]  # 该类的Dirichlet比例
        num_to_allocate = int(proportion * num_samples)  # 分配的样本数
        allocated_indices = indices[:num_to_allocate]
        client_indices.extend(allocated_indices)
    return client_indices
```

#### LDS的含义

**Label Distribution Skew（标签分布偏斜）**：
- 每个客户端对应一个域（Art, Clipart, Product, Real_World）
- 每个客户端可能接触到所有65个类，但每个类的**样本数量严重不均**
- alpha=0.1时，Dirichlet分布非常不均匀：
  - 某些类可能有大量样本
  - 某些类可能只有很少样本
  - 某些类可能完全没有样本（当 `int(proportion * num_samples) == 0`）

#### 数据流程

```
1. 对每个域（Art/Clipart/Product/Real_World）：
   - 加载该域的所有65个类的数据
   - 使用 train_ratio=0.7 划分训练集/测试集

2. 对每个类（0-64）：
   - 从该域的训练集中获取该类的所有样本
   - 根据Dirichlet比例 proportion[class_label] 计算要保留的样本数
   - num_to_allocate = int(proportion * num_samples)
   - 只保留前 num_to_allocate 个样本

3. 结果：
   - 每个客户端的训练数据中，各类样本数量极度不均
   - 这就是"Label Distribution Skew"
```

### 2. 当前FederatedScope实现的问题

#### 问题1: `splitter: lda_domain` 未实现

配置文件中设置了：
```yaml
splitter: lda_domain
splitter_args:
  - alpha: 0.1
```

但在 `federatedscope/cv/dataloader/dataloader.py` 中：
- **完全没有检查** `config.data.splitter` 参数
- 只检查 `dirichlet_alpha` 参数：
  ```python
  dirichlet_alpha = getattr(config.data, 'dirichlet_alpha', 0.0)
  ```

#### 问题2: Dirichlet实现方式错误

当前实现（`load_cv_dataset` 函数，第572-771行）：

```python
elif name == 'officehome':
    if desired_client_num <= num_domains:
        # 每个客户端分配一个完整的域
        for client_idx, domain in enumerate(domains[:desired_client_num], start=1):
            domain_data = load_office_home_domain_data(...)
            if dirichlet_alpha > 0:
                # 这里什么都没做！
                data_dict[client_idx] = domain_data
            else:
                data_dict[client_idx] = domain_data
```

**问题**：
- 当 `client_num=4` 时，每个客户端获得一个完整域的**所有数据**
- **没有按照Dirichlet分布对每个类的样本进行采样**
- 结果：每个客户端都有所有65个类的所有样本

#### 问题3: 与论文源码的差异对比

| 方面 | 论文源码 | 当前FederatedScope实现 |
|------|----------|----------------------|
| 数据分割方式 | 每个客户端从其域中按Dirichlet比例采样每个类的样本 | 每个客户端获得其域的所有样本 |
| 类别数量 | 每个客户端都可能有65个类，但样本数差异巨大 | 每个客户端都有65个类的所有样本 |
| LDS效果 | 强烈的标签分布偏斜（某些类几乎没有样本） | **无标签分布偏斜** |
| alpha参数 | 控制Dirichlet分布的集中度 | **未被使用** |

## 为什么导致低准确率？

### 根本原因

**数据分布不匹配导致GGEUR的数据增强无效**：

1. **论文设定（LDS）**：
   - 客户端A: 类0有100样本，类1有5样本，类2有0样本...
   - 客户端B: 类0有2样本，类1有80样本，类2有3样本...
   - **GGEUR的作用**：为样本少的类生成合成数据，平衡类别分布

2. **当前实现（无LDS）**：
   - 客户端A: 所有类都有大量样本
   - 客户端B: 所有类都有大量样本
   - **GGEUR的作用**：为本来就平衡的数据生成更多数据（意义不大）

3. **后果**：
   - GGEUR的数据增强策略在平衡数据上效果不佳
   - 生成的合成数据可能引入噪声而非帮助
   - 模型在初始阶段就表现差（5%准确率）

## 解决方案

### 方案1: 实现真正的LDS数据分割器

需要在 `load_cv_dataset` 函数中添加LDS逻辑：

```python
elif name == 'officehome':
    # 检查是否使用LDS
    use_lds = (config.data.splitter == 'lda_domain')

    if use_lds:
        # 获取alpha参数
        alpha = config.data.splitter_args[0].get('alpha', 0.1)

        # 生成4x65的Dirichlet矩阵
        dirichlet_matrix = np.random.dirichlet([alpha] * 4, 65).T

        # 对每个域应用Dirichlet采样
        for domain_idx, domain in enumerate(domains):
            domain_data = load_office_home_domain_data(...)

            # 对训练集应用Dirichlet采样
            train_dataset = domain_data['train']
            sampled_indices = apply_dirichlet_sampling(
                train_dataset,
                dirichlet_matrix[domain_idx],
                num_classes=65
            )
            domain_data['train'] = Subset(train_dataset, sampled_indices)

            data_dict[client_idx] = domain_data
    else:
        # 原有的实现（每个客户端一个完整域）
        ...
```

### 方案2: 修复配置文件

当前配置：
```yaml
splitter: lda_domain  # 未被使用！
splitter_args:
  - alpha: 0.1
```

应该添加：
```yaml
data:
  dirichlet_alpha: 0.1  # 当前实现会检查这个参数
  # 或者实现方案1，让splitter参数生效
```

### 方案3: 使用预处理好的LDS数据集

论文源码已经生成了LDS数据集，可以：
1. 运行论文源码的数据预处理脚本
2. 使用生成的 `Office-Home-0.1` 数据集
3. 修改配置文件指向预处理好的数据

## 推荐行动

### 立即执行

1. **运行诊断脚本**确认当前数据分布：
   ```bash
   python diagnose_officehome_distribution.py
   ```

   预期结果：所有客户端都有65个类，每个类都有大量样本

2. **检查论文源码生成的数据**（如果有）：
   ```bash
   ls ProjectsFederatedScopeGGEUR_source/Multi\ Domain/Office-Home-LDS/new_dataset/
   ```

### 后续修复

根据诊断结果选择方案：
- 如果确认当前无LDS → 实现方案1（推荐）或方案3
- 如果时间紧迫 → 方案3（使用预处理数据）
- 如果需要灵活性 → 方案1（实现LDS分割器）

## 预期改进

修复LDS后：
- 初始准确率应该提升（但不会太高，因为数据确实很不平衡）
- GGEUR的数据增强会真正发挥作用
- 训练过程中准确率会稳步提升
- 最终准确率应该接近论文报告的结果
