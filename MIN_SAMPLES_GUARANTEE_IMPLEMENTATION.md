# 最小样本数保证机制 - 实现文档

## 问题描述

在使用Dirichlet分布进行数据异构划分时（特别是alpha=0.1的极度异构情况），某些客户端可能被分配到0个训练样本，导致训练失败：

```
ValueError: No train_data or train_loader in the trainer
```

## 解决方案

实现了**最小样本数保证机制**，确保每个客户端至少获得指定数量的训练样本（默认5个），通过从样本充足的客户端重新分配样本来实现。

## 实现细节

### 1. 修改核心分割函数

**文件**: `federatedscope/cv/dataloader/dataloader.py:12-125`

**功能**: 在 `split_data_by_dirichlet` 函数中添加后处理逻辑：

```python
def split_data_by_dirichlet(dataset, num_clients, num_classes, alpha, seed,
                           min_samples_per_client=5):
    """
    新增参数:
        min_samples_per_client: 每个客户端必须拥有的最小样本数（默认5）
                               设为0可禁用此功能
    """
    # ... 原有的Dirichlet分布逻辑 ...

    # 后处理：确保最小样本数
    if min_samples_per_client > 0:
        # 1. 识别需要样本的客户端（< min_samples_per_client）
        clients_need_samples = [...]

        # 2. 识别有额外样本的客户端（> min_samples_per_client * 2）
        clients_have_extra = [...]

        # 3. 从有余量的客户端转移样本到不足的客户端
        for client_id in clients_need_samples:
            needed = min_samples_per_client - len(client_indices[client_id])
            # 从donors收集样本
            for donor_id in clients_have_extra:
                can_donate = max(0, len(donor_samples) - min_samples_per_client)
                # 转移样本...
```

**算法特点**：
- 只从样本数 > 2×最小值的客户端"借"样本
- 确保donor在捐赠后仍然保有至少最小样本数
- 保持数据异构性，只做最小调整

### 2. 添加配置参数

**文件**: `federatedscope/core/configs/cfg_data.py:32-34`

```python
cfg.data.min_samples_per_client = 5  # 最小样本数
# 每个客户端在Dirichlet分割后必须拥有的最小训练样本数
# 样本会从有额外样本的客户端自动重新分配
# 设为0可禁用此功能
```

### 3. 更新数据加载器调用

**文件**: `federatedscope/cv/dataloader/dataloader.py:298-306 & 496-504`

```python
# 获取配置参数
min_samples = getattr(config.data, 'min_samples_per_client', 5)

# 调用分割函数（仅对训练集启用）
client_indices_list = split_data_by_dirichlet(
    dataset=dataset,
    num_clients=num_clients_this_domain,
    num_classes=num_classes,
    alpha=dirichlet_alpha,
    seed=config.seed + domain_id + hash(split_name) % 1000,
    min_samples_per_client=min_samples if split_name == 'train' else 0
)
```

**注意**：只对训练集启用最小样本保证，验证集和测试集不做限制。

### 4. 更新PACS配置

**文件**: `scripts/example_configs/cross_domain_adaptive_pacs.yaml:22`

```yaml
data:
  root: '/root/data/PACS'
  type: 'pacs'
  splits: [0.8, 0.1, 0.1]
  dirichlet_alpha: 0.1
  min_samples_per_client: 5  # 新增：最小训练样本数
  server_test_samples_per_class: 40
```

## 测试验证

### 测试场景
- 2576个训练样本，7个类，25个客户端
- Dirichlet alpha = 0.1（极度异构）
- 最小样本数 = 5

### 测试结果

**没有最小样本保证**：
```
Min samples: 0
Clients with 0 samples: 1 / 25 (4.0%)
Clients with < 5 samples: 3 / 25 (12.0%)
Distribution: [0, 1, 2, 10, 14, ..., 275]
```

**有最小样本保证**：
```
Found 3 clients with < 5 samples
  Client 4: 2 -> 5 samples (added 3)
  Client 21: 1 -> 5 samples (added 4)
  Client 23: 0 -> 5 samples (added 5)

Min samples: 5
Clients with 0 samples: 0 / 25 (0.0%)
Clients with < 5 samples: 0 / 25 (0.0%)
Distribution: [5, 5, 5, 10, 14, ..., 275]
```

**✓ 成功**：所有客户端都至少有5个训练样本！

## 使用方法

### 方法1：使用默认值（推荐）

配置文件中不需要额外设置，默认每个客户端至少5个样本：

```yaml
data:
  dirichlet_alpha: 0.1
  # min_samples_per_client 默认为 5
```

### 方法2：自定义最小样本数

根据需求调整：

```yaml
data:
  dirichlet_alpha: 0.1
  min_samples_per_client: 10  # 更严格的最小值
```

### 方法3：禁用此功能

如果需要原始的Dirichlet分布（允许0样本客户端）：

```yaml
data:
  dirichlet_alpha: 0.1
  min_samples_per_client: 0  # 禁用
```

## 原始配置对比

### 修改前的问题配置
```yaml
federate:
  client_num: 100
data:
  dirichlet_alpha: 0.1
  # 没有 min_samples_per_client
```
**结果**：某些客户端得到0样本 → 训练失败

### 修改后的工作配置
```yaml
federate:
  client_num: 100
data:
  dirichlet_alpha: 0.1
  min_samples_per_client: 5  # 新增
```
**结果**：所有客户端至少5样本 → 训练成功

## 优势

1. **保持原有参数**：不需要减少客户端数量或增加alpha值
2. **保持异构性**：只做最小必要的调整，大部分分布保持不变
3. **自动化处理**：无需手动过滤空客户端
4. **灵活配置**：可根据场景调整最小值或禁用
5. **日志可见**：清楚记录哪些客户端接收了额外样本

## 日志输出示例

运行时会看到类似日志：

```
INFO: Splitting domain 'photo' among 25 clients using Dirichlet distribution (alpha=0.1)
INFO: Minimum samples per client: 5 (train only)
INFO: Found 3 clients with < 5 samples
INFO:   Client 4: 2 -> 5 samples (added 3)
INFO:   Client 21: 1 -> 5 samples (added 4)
INFO:   Client 23: 0 -> 5 samples (added 5)
```

## 注意事项

1. **仅适用于训练集**：验证集和测试集不受影响（min_samples设为0）
2. **需要足够的总样本数**：如果总样本数 < (min_samples × num_clients)，某些客户端仍可能不足
3. **donor限制**：只从样本数 > 2×min_samples的客户端转移，避免过度均衡化
4. **适用范围**：Office-Caltech和PACS数据集的Dirichlet分割都已启用

## 文件修改列表

1. ✓ `federatedscope/cv/dataloader/dataloader.py` - 核心分割逻辑
2. ✓ `federatedscope/core/configs/cfg_data.py` - 配置参数定义
3. ✓ `scripts/example_configs/cross_domain_adaptive_pacs.yaml` - PACS配置更新
4. ✓ `test_min_samples_guarantee.py` - 功能测试脚本

## 服务器部署

直接运行即可，无需修改原有配置中的client_num或dirichlet_alpha：

```bash
python federatedscope/main.py --cfg scripts/example_configs/cross_domain_adaptive_pacs.yaml
```

系统会自动确保每个客户端至少有5个训练样本，解决"No train_data"错误。
