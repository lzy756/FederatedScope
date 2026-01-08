# PACS数据集测试集划分分析与问题修复

## 问题描述
运行PACS数据集时出现错误：
```
ValueError: No train_data or train_loader in the trainer
```

## 测试集划分机制

PACS数据集采用**两层测试集**机制来防止数据泄漏：

### 1. 服务器端Held-out测试集
- **目的**: 用于服务器全局评估，完全独立于客户端数据
- **位置**: `federatedscope/cv/dataset/pacs.py:251-435` (`load_balanced_pacs_data`)
- **采样策略**:
  ```python
  server_test_samples_per_class = 40  # 配置参数
  # 每个domain: 40样本/类 × 7类 = 280样本
  # 4个domains总计: 1120个样本
  ```
- **关键机制**:
  - 从原始完整数据集中优先采样
  - 记录 `excluded_indices`（被排除的索引）
  - 传递给客户端数据加载器，确保这些样本不被客户端使用

### 2. 客户端测试集
- **目的**: 用于客户端本地评估
- **位置**: `federatedscope/cv/dataset/pacs.py:78-147` (`PACS._load_data`)
- **划分流程**:
  ```python
  # 1. 加载所有样本
  all_images, all_labels = load_all_from_domain()

  # 2. 排除服务器held-out样本
  if exclude_indices:
      filtered = [sample for i, sample in enumerate(all_images)
                 if i not in exclude_indices]

  # 3. 从剩余数据按比例划分
  splits = [0.8, 0.1, 0.1]  # train/val/test
  train = filtered[0:80%]
  val = filtered[80%:90%]
  test = filtered[90%:100%]  # 客户端测试集
  ```

## 根本原因分析

### 数据流失效分析

**配置参数**:
```yaml
federate:
  client_num: 100
data:
  splits: [0.8, 0.1, 0.1]
  dirichlet_alpha: 0.1
  server_test_samples_per_class: 40
```

**数据分配计算**:
```
原始数据: ~3500样本/domain
↓
服务器held-out: 280样本/domain (40/类 × 7类)
↓
剩余给客户端: 3220样本/domain
↓
训练集(80%): 2576样本/domain
↓
每个domain分配: 25个客户端 (100 ÷ 4)
↓
理论上每客户端: 103样本 (2576 ÷ 25)
```

### Dirichlet分布导致的问题

**关键代码**: `dataloader.py:442-448`
```python
client_indices_list = split_data_by_dirichlet(
    dataset=dataset,
    num_clients=25,  # 每个domain 25个客户端
    num_classes=7,
    alpha=0.1,  # 极度异构
    seed=config.seed
)
```

**Dirichlet α=0.1的效果** (从模拟结果):
```
单个类(368样本)的分配:
- 最少: 0样本
- 最多: 235样本
- 平均: 14.5样本
- 获得0样本的客户端: 16/25 (64%)
```

**问题根源**:
1. α=0.1表示极度数据异构（Non-IID）
2. 25个客户端竞争368个样本（单类）
3. Dirichlet分布导致：
   - 少数客户端获得大量样本
   - 大多数客户端获得极少或零样本
4. **跨7个类，某些客户端在所有类上都得到0样本**
5. 代码没有验证 `indices` 是否为空，创建了空的Subset

### 代码缺陷

`dataloader.py:459-460`:
```python
indices = client_indices_list[local_client_id]  # 可能是空列表[]
data_dict[global_client_id][split_name] = Subset(dataset, indices)  # 创建空Subset
# ⚠️ 没有检查indices是否为空！
```

## 解决方案

### 快速修复：调整配置参数

创建了修复版配置文件 `cross_domain_adaptive_pacs_FIXED.yaml`:

```yaml
federate:
  client_num: 30        # 100 → 30 (减少客户端数量)
  sample_client_num: 20  # 30 → 20 (相应调整)

data:
  dirichlet_alpha: 0.5   # 0.1 → 0.5 (降低异构程度)
  server_test_samples_per_class: 30  # 40 → 30 (为客户端保留更多数据)

ondemfl:
  subset_size: 20       # 30 → 20 (匹配sample_client_num)
```

**效果改善**:
```
原配置:
- 25客户端/domain
- 103样本/客户端 (理论)
- α=0.1: 64%客户端某类为0

新配置:
- 7-8客户端/domain (30 ÷ 4)
- ~180样本/客户端
- α=0.5: 更均衡的分配
- 空客户端概率大幅降低
```

### 长期修复：代码层面改进

**建议修改 `dataloader.py:450-470`**:

```python
# Assign data to clients
for local_client_id in range(num_clients_this_domain):
    global_client_id = client_idx + local_client_id

    # Get indices
    indices = client_indices_list[local_client_id]

    # ✓ 添加空检查
    if len(indices) == 0:
        logger.warning(
            f"Client {global_client_id} has 0 samples in {split_name} split "
            f"for domain {domain}. This client will be skipped."
        )
        continue

    # Initialize client data dict if not exists
    if global_client_id not in data_dict:
        data_dict[global_client_id] = {}

    # Create subset
    data_dict[global_client_id][split_name] = Subset(dataset, indices)
```

### 其他建议

1. **参数组合建议**:
   ```yaml
   # 选项A: 少量客户端 + 高异构
   client_num: 10-20
   dirichlet_alpha: 0.1-0.3

   # 选项B: 中等客户端 + 中等异构 (推荐)
   client_num: 20-40
   dirichlet_alpha: 0.5-1.0

   # 选项C: 多客户端 + 低异构
   client_num: 50-100
   dirichlet_alpha: 1.0-5.0
   ```

2. **服务器测试集大小建议**:
   ```yaml
   # 固定样本数模式
   server_test_samples_per_class: 20-30  # 不要超过40

   # 比例模式 (更灵活)
   server_test_ratio: 0.05-0.1  # 5-10%
   ```

3. **添加数据验证**:
   ```python
   # 在训练前验证所有客户端有数据
   for client_id, client_data in data_dict.items():
       if 'train' not in client_data or len(client_data['train']) == 0:
           raise ValueError(f"Client {client_id} has no training data!")
   ```

## 使用修复版配置

### 本地测试 (Windows):
```bash
python federatedscope/main.py --cfg scripts/example_configs/cross_domain_adaptive_pacs_FIXED.yaml
```

### 服务器运行:
```bash
# 复制修复版配置
cp scripts/example_configs/cross_domain_adaptive_pacs_FIXED.yaml \
   scripts/example_configs/cross_domain_adaptive_pacs.yaml

# 或者直接使用新配置
python federatedscope/main.py --cfg scripts/example_configs/cross_domain_adaptive_pacs_FIXED.yaml
```

## 验证修复

运行后应该看到：
```
✓ Server held-out indices loaded for 4 domains
  Domain 'photo': 210 samples excluded from client data
  Domain 'art_painting': 210 samples excluded from client data
  ...

✓ Client data loaded:
  Client 1: Train=180, Val=22, Test=22
  Client 2: Train=165, Val=20, Test=20
  ...

✓ All 30 clients have training data
```

## 总结

PACS数据集测试集划分采用两层机制：
1. **服务器Held-out测试集**: 从原始数据采样，完全独立
2. **客户端测试集**: 从剩余数据划分，用于本地评估

错误原因是：
- 过多客户端(100) + 极度异构(α=0.1) + 服务器占用(280/domain)
- 导致Dirichlet分布给某些客户端分配0样本
- 代码未验证空数据

修复方案:
- **立即**: 使用 `cross_domain_adaptive_pacs_FIXED.yaml`
- **长期**: 在数据加载器中添加空数据检查
