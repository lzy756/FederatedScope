# 🎯 找到根本原因！num_test_epoch 缺失

## 问题确认

从调试日志看到：
```
[DEBUG _hook_on_epoch_start] Reset existing ReIterator
WARNING: No data for evaluation in split 'test'
```

**关键观察:**
- `_hook_on_epoch_start` 执行了 ✓
- `_hook_on_batch_start_init` **完全没有执行** ✗
- 直接跳到 "No data" 警告

## 根本原因

在 `federatedscope/core/trainers/trainer.py:287-288`：

```python
def _run_epoch(self, hooks_set):
    for epoch_i in range(getattr(self.ctx, f"num_{self.ctx.cur_split}_epoch")):
        # ... epoch 循环
```

当 `cur_split='test'` 时，它查找 `ctx.num_test_epoch`。

**如果 `num_test_epoch` 是 0 或不存在，`range(0)` 返回空迭代器，循环不执行！**

这就是为什么：
- epoch_start 钩子执行（在循环外）
- 批次循环完全不执行（在循环内）
- 没有数据被处理

## 最终修复

**文件:** `federatedscope/contrib/worker/cross_domain_adaptive.py`
**位置:** 第 467-470 行

```python
# CRITICAL: Set num_test_epoch to ensure evaluation loop runs
# The trainer's _run_epoch uses this to determine loop iterations
if not hasattr(self.trainer.ctx, 'num_test_epoch') or self.trainer.ctx.num_test_epoch == 0:
    self.trainer.ctx.num_test_epoch = 1
```

**同时添加了日志：**
```python
logger.info(f"  - num_test_epoch: {self.trainer.ctx.get('num_test_epoch')}")
```

## 为什么会缺失

服务器的 trainer 通常不执行评估，所以 `num_test_epoch` 可能：
1. 从未被初始化
2. 被设置为 0（表示不评估）
3. 在某个地方被清除了

客户端的 trainer 有正确的配置，但服务器端的 trainer 是我们手动调用的，需要手动设置。

## 验证修复

重新运行训练后，应该看到：

```
INFO: Domain amazon:
INFO:   - test_data set: True
INFO:   - test_loader set: True
INFO:   - test_data size: 100
INFO:   - test_loader batches: 25
INFO:   - cur_split: test
INFO:   - num_test_epoch: 1  ← 新增！应该是 1
INFO:   - check_split result: True

INFO: [DEBUG _hook_on_epoch_start] Reset existing ReIterator
INFO: [DEBUG _hook_on_batch_start_init] Attempting to get next batch  ← 应该看到这个！
INFO: [DEBUG _hook_on_batch_start_init] Successfully got batch
# ↑ 重复 25 次

# 不应再有 "No data" 警告
```

## 期望结果

1. ✅ 不再有 "No data for evaluation" 警告
2. ✅ 看到 25 次 "Successfully got batch"
3. ✅ 准确率开始更新（初期可能仍接近 0.1，随机水平）
4. ✅ 随着训练进行，准确率应该提升

## 如果仍然为 0

如果修复后：
- ✅ 没有 "No data" 警告
- ✅ 看到批次被处理
- ❌ 准确率仍然是 0

那么问题转移到：
1. **模型预测问题** - 检查模型输出
2. **标签问题** - 检查标签范围
3. **训练问题** - 检查损失是否下降

但至少数据遍历问题会被解决！

## 总结

**问题:** `num_test_epoch` 未设置或为 0
**影响:** `_run_epoch` 的循环不执行
**修复:** 手动设置 `ctx.num_test_epoch = 1`
**文件:** cross_domain_adaptive.py (第 467-470 行)

**这应该是最后一个阻碍数据遍历的问题！** 🎯
