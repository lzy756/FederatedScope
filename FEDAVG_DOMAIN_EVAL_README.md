# FedAvg Baseline with Domain-Specific Evaluation

## 概述

这个配置文件提供了一个标准的 FedAvg 基线实验，具有与 `cross_domain_adaptive` 方法相同的域特定评估逻辑，用于公平比较。

## 文件

### 配置文件
- **配置**: `scripts/example_configs/fedavg_office_caltech_domain_eval.yaml`
- **Server代码**: `federatedscope/contrib/worker/fedavg_domain_eval.py`

### 主要特性

1. **标准 FedAvg 训练**
   - 使用经典的 FedAvg 聚合方法
   - 没有任何特殊的域自适应技术
   - 作为对比实验的基线

2. **域特定评估**
   - 在每个域（amazon, webcam, dslr, caltech）上分别评估
   - 使用平衡的测试集（每个类别相同数量的样本）
   - 计算加权平均准确率
   - 与 cross_domain_adaptive 使用相同的评估逻辑

3. **相同的实验设置**
   - 数据集：Office-Caltech-10
   - 模型：fedlsa_cnn (512维隐藏层)
   - 训练参数：与 cross_domain_adaptive 完全一致
   - Dirichlet alpha: 0.1 (相同的数据异构性)

## 配置对比

### FedAvg Domain Eval vs Cross-Domain Adaptive

| 配置项 | FedAvg Domain Eval | Cross-Domain Adaptive |
|--------|-------------------|----------------------|
| 联邦方法 | `fedavg_domain_eval` | `cross_domain_adaptive` |
| 模型类型 | `fedlsa_cnn` | `cross_domain_adaptive` |
| Trainer | `general` | `cross_domain_adaptive` |
| FedLSA | ❌ 不使用 | ✅ 使用语义锚点对齐 |
| OnDemFL | ❌ 不使用 | ✅ 使用按需客户端选择 |
| 域特定评估 | ✅ 相同逻辑 | ✅ 相同逻辑 |
| 训练轮数 | 300 | 300 |
| 本地更新步数 | 8 | 8 |
| 学习率 | 0.01 | 0.01 |

## 使用方法

### 基本用法

```bash
python federatedscope/main.py \
  --cfg scripts/example_configs/fedavg_office_caltech_domain_eval.yaml
```

### 修改训练轮数

```bash
python federatedscope/main.py \
  --cfg scripts/example_configs/fedavg_office_caltech_domain_eval.yaml \
  federate.total_round_num 100
```

### 快速测试

```bash
python federatedscope/main.py \
  --cfg scripts/example_configs/fedavg_office_caltech_domain_eval.yaml \
  federate.total_round_num 10 \
  train.local_update_steps 2
```

## 输出示例

训练过程中，每个轮次结束后会看到类似以下的域特定评估结果：

```
================================================================================
[Round 10] Server-side Evaluation on Balanced Test Sets
================================================================================
--------------------------------------------------------------------------------
Domain-specific Test Results:
--------------------------------------------------------------------------------
  amazon      : Acc = 0.3250 | Samples =  100 | Weight = 0.259
  caltech     : Acc = 0.4111 | Samples =   90 | Weight = 0.233
  dslr        : Acc = 0.3878 | Samples =   98 | Weight = 0.254
  webcam      : Acc = 0.3960 | Samples =  100 | Weight = 0.259
--------------------------------------------------------------------------------
  Weighted Avg: Acc = 0.3777 | Total Samples =  388
================================================================================
```

## 实现细节

### Server 实现 (FedAvgDomainEvalServer)

`federatedscope/contrib/worker/fedavg_domain_eval.py` 中的 `FedAvgDomainEvalServer` 类：

1. **继承关系**
   - 继承自标准的 `Server` 类
   - 不依赖 FedLSA 或 OnDemFL 的任何组件

2. **核心方法**
   - `_load_balanced_test_data()`: 加载每个域的平衡测试集
   - `eval()`: 重写评估方法以执行域特定评估
   - `_restore_trainer_context()`: 恢复trainer上下文
   - `_log_domain_results()`: 记录和计算加权平均结果

3. **评估流程**
   ```
   对于每个域 (amazon, webcam, dslr, caltech):
     1. 临时设置该域的测试数据
     2. 设置 num_test_epoch 和 num_test_batch (关键！)
     3. 运行 trainer.evaluate()
     4. 收集域特定结果
     5. 恢复原始上下文

   计算所有域的加权平均准确率
   ```

### 关键技术点

1. **num_test_batch 设置**
   ```python
   # 必须设置这两个值，否则评估循环不会运行
   self.trainer.ctx.num_test_epoch = 1
   self.trainer.ctx.num_test_batch = len(test_loader)
   ```

2. **ReIterator 包装**
   ```python
   # DataLoader 必须用 ReIterator 包装以兼容 trainer
   from federatedscope.core.auxiliaries.ReIterator import ReIterator
   self.trainer.ctx.test_loader = ReIterator(test_loader)
   ```

3. **上下文管理**
   - 在评估每个域前保存原始上下文
   - 评估后必须恢复，避免影响其他域或训练

## 对比实验建议

### 运行 FedAvg baseline
```bash
python federatedscope/main.py \
  --cfg scripts/example_configs/fedavg_office_caltech_domain_eval.yaml \
  outdir exp_comparison/fedavg
```

### 运行 Cross-Domain Adaptive
```bash
python federatedscope/main.py \
  --cfg scripts/example_configs/cross_domain_adaptive_office_caltech.yaml \
  outdir exp_comparison/cross_domain_adaptive
```

### 对比指标

主要关注以下指标：
- **域特定准确率**: 每个域的测试准确率
- **加权平均准确率**: 所有域的加权平均（按样本数加权）
- **收敛速度**: 达到特定准确率所需的轮数
- **稳定性**: 准确率的方差

## 注意事项

1. **数据路径**
   - 确保修改配置中的 `data.root` 为实际的数据路径
   - 默认路径: `/root/data/office_caltech_10`

2. **GPU 使用**
   - 如需使用 GPU，确保 `use_gpu: True` 和 `device: 0`
   - 如使用 CPU，设置 `use_gpu: False`

3. **随机种子**
   - 为了可重复性，两个方法使用相同的 seed: 123
   - 可以修改 `seed` 参数进行多次实验

4. **评估频率**
   - 当前配置每轮都评估 (`eval.freq: 1`)
   - 可以调整以减少评估开销

## 扩展

### 添加其他基线方法

可以参考这个实现为其他方法（如 FedProx, FedOpt 等）添加域特定评估：

1. 创建新的 server 类，继承对应的基类
2. 添加 `_load_balanced_test_data()` 和自定义 `eval()` 方法
3. 注册 worker
4. 创建配置文件

### 修改评估逻辑

如需修改评估方式（如使用不同的测试集、评估指标等），主要修改 `FedAvgDomainEvalServer.eval()` 方法。

## 常见问题

**Q: 为什么准确率全为0？**

A: 检查以下几点：
- 确保设置了 `num_test_epoch` 和 `num_test_batch`
- 确认 test_loader 已正确包装为 ReIterator
- 查看日志确认评估循环是否运行

**Q: 如何只使用标准 FedAvg 而不要域评估？**

A: 使用现有的 `scripts/example_configs/fedavg_office_caltech.yaml` 配置文件，它使用标准的 FedAvg 而不包含域特定评估。

**Q: 可以用于其他数据集吗？**

A: 当前实现专门针对 Office-Caltech 数据集。如需支持其他数据集，需要：
1. 在 `_load_balanced_test_data()` 中添加相应的数据加载逻辑
2. 或者修改为更通用的实现

## 参考

- Cross-Domain Adaptive 实现: `federatedscope/contrib/worker/cross_domain_adaptive.py`
- FedAvg 原始论文: McMahan et al., "Communication-Efficient Learning of Deep Networks from Decentralized Data"
- Office-Caltech-10 数据集: Gong et al., "Geodesic Flow Kernel for Unsupervised Domain Adaptation"
