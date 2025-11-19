# 服务器端评估"无数据"问题调试指南

## 当前问题

训练能运行，测试集分配到服务器，但出现以下警告：
```
WARNING: No data for evaluation in split 'test', returning zero metrics: ['test_total', 'test_acc', 'test_loss', 'test_avg_loss']
```

这表明评估时没有实际遍历任何数据批次。

## 已完成的修复

### 1. 添加 `test_total` 键（✓ 完成）
**文件:** `federatedscope/contrib/worker/cross_domain_adaptive.py`
- 第 478-481 行: 添加 `test_total` 到域评估结果
- 第 578 行: 添加 `test_total` 到加权平均结果

### 2. 添加调试日志（✓ 完成）
**文件:** `federatedscope/contrib/worker/cross_domain_adaptive.py`
**位置:** 第 462-467 行

```python
# Debug: verify data was set and check loader
logger.info(f"Domain {domain}:")
logger.info(f"  - test_data set: {self.trainer.ctx.get('test_data') is not None}")
logger.info(f"  - test_loader set: {self.trainer.ctx.get('test_loader') is not None}")
logger.info(f"  - test_data size: {len(self.trainer.ctx.get('test_data', []))}")
logger.info(f"  - test_loader batches: {len(test_loader)}")
```

## 下一步诊断

### 步骤 1: 重新运行训练

```bash
python federatedscope/main.py --cfg scripts/example_configs/cross_domain_adaptive_office_caltech.yaml
```

### 步骤 2: 检查新的调试日志

查找服务器端评估的日志输出，应该看到类似：

```
INFO: Evaluating domain: amazon
INFO: Domain amazon:
INFO:   - test_data set: True/False
INFO:   - test_loader set: True/False
INFO:   - test_data size: XXX
INFO:   - test_loader batches: XXX
```

**关键检查点:**
1. **test_data set** 应该是 `True`
2. **test_loader set** 应该是 `True`
3. **test_data size** 应该是 100 (每域 10 类 × 10 样本)
4. **test_loader batches** 应该 > 0 (例如 25，如果 batch_size=4)

### 步骤 3: 根据日志诊断问题

#### 情况 A: test_data/test_loader 都是 False
**原因:** 数据没有被正确加载到服务器
**解决:**
1. 检查 `_load_balanced_test_data()` 是否被调用
2. 检查是否有加载错误日志
3. 运行 `python diagnose_server_test_data.py` 验证数据

#### 情况 B: test_data/test_loader 是 True，但 size=0 或 batches=0
**原因:** 数据加载了但是空的
**解决:**
1. 检查数据路径是否正确
2. 检查 `samples_per_class` 配置
3. 检查数据集是否真的有图片

#### 情况 C: test_data/test_loader 都是 True，size > 0，batches > 0
**原因:** 数据正确加载，但评估循环没有遍历
**可能原因:**
1. `check_split('test')` 返回 False
2. 评估循环内部有异常
3. Trainer 的 hooks 有问题

## 可能的根本原因

### 原因 1: Context 数据设置问题

`check_split` 方法检查 `ctx.get('test_data')` 和 `ctx.get('test_loader')`。如果返回 None，评估会被跳过。

**验证:**
```python
# 添加到 cross_domain_adaptive.py evaluate() 方法中
logger.info(f"check_split('test') = {self.trainer.ctx.check_split('test', skip=True)}")
```

### 原因 2: DataLoader 被包装或重置

在 `_hook_on_epoch_start` (torch_trainer.py:246)，如果 loader 不是 `ReIterator`，它会被重新包装：

```python
if ctx.get("{}_loader".format(ctx.cur_split)) is None:
    loader = get_dataloader(
        WrapDataset(ctx.get("{}_data".format(ctx.cur_split))),
        self.cfg, ctx.cur_split)
    setattr(ctx, "{}_loader".format(ctx.cur_split), ReIterator(loader))
```

**可能问题:**
- WrapDataset 可能不兼容我们的数据集
- ReIterator 包装可能导致问题

### 原因 3: 评估循环内部异常

评估可能在循环内部遇到异常，导致 `StopIteration` 或其他错误。

## 建议的修复方案

### 方案 A: 确保 ReIterator 兼容

在设置 test_loader 时，直接使用 ReIterator 包装：

```python
from federatedscope.core.auxiliaries.ReIterator import ReIterator

# 在 cross_domain_adaptive.py 中
self.trainer.ctx.test_data = test_loader.dataset
self.trainer.ctx.test_loader = ReIterator(test_loader)
```

### 方案 B: 使用 ClientData 结构

将测试数据包装成 ClientData 格式：

```python
from federatedscope.core.data import ClientData

test_data_dict = {
    'test': test_loader.dataset
}
self.trainer.ctx.data = test_data_dict
self.trainer.ctx.test_data = test_loader.dataset
```

### 方案 C: 直接调用 _run_routine

绕过 evaluate() 方法，直接调用 _run_routine：

```python
from federatedscope.core.trainers.enums import MODE

# 设置数据
self.trainer.ctx.test_data = test_loader.dataset
self.trainer.ctx.test_loader = test_loader
self.trainer.ctx.cur_split = 'test'

# 直接运行评估循环
self.trainer._run_routine(MODE.TEST, self.trainer.hooks_in_eval, 'test')

# 获取结果
test_metrics = self.trainer.ctx.eval_metrics
```

## 测试脚本

运行以下脚本验证 Context 和 DataLoader 基本功能：

```bash
python test_server_eval_data.py
```

## 需要的信息

请重新运行训练并提供以下信息：

1. **调试日志输出:**
   ```
   Domain amazon:
     - test_data set: ?
     - test_loader set: ?
     - test_data size: ?
     - test_loader batches: ?
   ```

2. **是否有其他错误或警告？**

3. **客户端本地训练准确率是否在提升？**
   - 如果客户端准确率正常提升，说明模型在学习
   - 只是服务器端评估有问题

## 快速修复尝试

如果上述调试还不能解决，可以尝试这个快速修复：

**文件:** `federatedscope/contrib/worker/cross_domain_adaptive.py`
**位置:** 第 459-461 行

```python
# 当前代码
self.trainer.ctx.test_data = test_loader.dataset
self.trainer.ctx.test_loader = test_loader

# 修改为
from federatedscope.core.auxiliaries.ReIterator import ReIterator

self.trainer.ctx.test_data = test_loader.dataset
# 直接包装成 ReIterator
self.trainer.ctx.test_loader = ReIterator(test_loader)
# 确保 cur_split 被设置
self.trainer.ctx.cur_split = 'test'
```

这样可以确保 loader 格式与 trainer 期望的一致。

## 总结

当前的主要问题是评估循环没有遍历数据。通过添加的调试日志，我们可以确定：
1. 数据是否被正确加载和设置
2. Context 中是否能访问到数据
3. DataLoader 是否有批次

根据日志输出，我们可以进一步定位问题并应用相应的修复方案。

**请重新运行训练并分享日志中的调试信息！**
