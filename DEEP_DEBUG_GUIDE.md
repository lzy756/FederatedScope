# 深度调试：评估循环不遍历数据

## 当前状态

✅ check_split result: True - 数据设置正确
✅ test_loader batches: 23 - 有批次数据
❌ 仍然有 "No data for evaluation" 警告

## 添加的深度调试

在 `torch_trainer.py` 中添加了详细的调试日志来追踪评估循环的执行：

### 1. `_hook_on_epoch_start` 调试（第 249-270 行）

显示 DataLoader 是如何被处理的：
- existing_loader 是否为 None
- existing_loader 的类型
- 是否是 ReIterator
- 执行了哪个分支（创建/包装/重置）

### 2. `_hook_on_batch_start_init` 调试（第 286-300 行）

显示批次获取的过程：
- 尝试获取批次
- loader 类型
- 是否成功获取
- 是否遇到 StopIteration

## 重新运行并查找

```bash
python federatedscope/main.py --cfg scripts/example_configs/cross_domain_adaptive_office_caltech.yaml
```

### 关键日志模式

应该看到类似的输出序列：

```
INFO: Domain caltech:
INFO:   - test_data set: True
INFO:   - test_loader set: True
INFO:   - test_data size: 90
INFO:   - test_loader batches: 23
INFO:   - cur_split: test
INFO:   - check_split result: True

# ← 然后应该看到这些调试日志：
INFO: [DEBUG _hook_on_epoch_start] cur_split=test
INFO: [DEBUG _hook_on_epoch_start] existing_loader is None: False
INFO: [DEBUG _hook_on_epoch_start] existing_loader type: <class 'xxx'>
INFO: [DEBUG _hook_on_epoch_start] is ReIterator: True/False
INFO: [DEBUG _hook_on_epoch_start] Reset existing ReIterator  # 或其他分支

INFO: [DEBUG _hook_on_batch_start_init] Attempting to get next batch
INFO: [DEBUG _hook_on_batch_start_init] loader type: <class 'xxx'>
INFO: [DEBUG _hook_on_batch_start_init] Successfully got batch  # 或 StopIteration

# 应该重复 23 次（对应 23 个批次）
```

## 诊断指南

### 情况 A: 看不到任何 DEBUG 日志
**问题:** 评估循环根本没有执行
**原因:** `check_split` 之后，`trainer.evaluate()` 可能立即返回
**解决:** 需要检查 `trainer.evaluate()` 方法本身

### 情况 B: 看到 _hook_on_epoch_start，但没有 _hook_on_batch_start_init
**问题:** epoch 开始了，但没有尝试获取批次
**原因:** epoch 循环可能立即结束，没有进入批次循环
**解决:** 需要检查 `_run_epoch` 方法

### 情况 C: 看到 _hook_on_batch_start_init，但立即 StopIteration
**问题:** 尝试获取批次但立即失败
**原因:**
- ReIterator 没有正确初始化
- DataLoader 是空的或已耗尽
- 包装的 loader 类型不对

**解决:**
- 检查 existing_loader type
- 检查 is ReIterator
- 可能需要调整 ReIterator 包装方式

### 情况 D: Successfully got batch 但仍为 0
**问题:** 批次成功获取，但准确率仍为 0
**原因:**
- 模型还未训练（初期正常）
- 模型预测有问题
- 标签或数据有问题

**解决:** 这可能是正常的，继续观察几轮

## 可能的根本问题

基于 check_split=True 但仍无数据，最可能的原因是：

1. **ReIterator 包装问题**
   - DataLoader 被包装了两次
   - ReIterator 的 reset() 方法有问题
   - isinstance 检查失败

2. **Epoch 循环配置**
   - num_epochs 设置为 0
   - 某个配置导致循环提前退出

3. **Batch 迭代器**
   - StopIteration 在第一次就抛出
   - DataLoader 的 __iter__ 方法有问题

## 下一步

**请运行训练并分享 DEBUG 日志**，特别是：

1. `[DEBUG _hook_on_epoch_start]` 的所有输出
2. `[DEBUG _hook_on_batch_start_init]` 的所有输出
3. 是否看到多次批次日志（应该有 23 次）

这些日志会告诉我们评估循环在哪里卡住了！
