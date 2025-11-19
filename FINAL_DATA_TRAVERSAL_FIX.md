# 服务器端评估数据遍历修复（最终版本）

## 问题

虽然不再报错，但服务器端评估仍然显示：
```
WARNING: No data for evaluation in split 'test', returning zero metrics
```

所有域的准确率都是 0，说明评估循环没有实际处理任何数据批次。

## 根本原因

trainer 的 `_hook_on_epoch_start` 方法会检查并包装 DataLoader：
1. 如果 loader 不是 `ReIterator` 类型，会被重新包装
2. 我们直接传入的 PyTorch `DataLoader` 需要被包装成 `ReIterator`
3. `cur_split` 需要被正确设置，否则 `check_split` 会失败

## 最终修复

**文件:** `federatedscope/contrib/worker/cross_domain_adaptive.py`

### 修改 1: 数据设置（第 453-481 行）

```python
# Temporarily set the test data for this domain
original_test_data = self.trainer.ctx.get('test_data')
original_test_loader = self.trainer.ctx.get('test_loader')
original_data = self.trainer.ctx.get('data')
original_cur_split = self.trainer.ctx.get('cur_split')  # ← 新增

# Set test data with proper structure
# Use ReIterator to wrap the loader for compatibility with trainer
from federatedscope.core.auxiliaries.ReIterator import ReIterator

self.trainer.ctx.test_data = test_loader.dataset
self.trainer.ctx.test_loader = ReIterator(test_loader)  # ← ReIterator 包装
self.trainer.ctx.cur_split = 'test'  # ← 设置 cur_split

# Debug: verify data was set and check loader
logger.info(f"Domain {domain}:")
logger.info(f"  - test_data set: {self.trainer.ctx.get('test_data') is not None}")
logger.info(f"  - test_loader set: {self.trainer.ctx.get('test_loader') is not None}")
logger.info(f"  - test_data size: {len(self.trainer.ctx.get('test_data', []))}")
logger.info(f"  - test_loader batches: {len(test_loader)}")
logger.info(f"  - cur_split: {self.trainer.ctx.get('cur_split')}")
logger.info(f"  - check_split result: {self.trainer.ctx.check_split('test', skip=True)}")  # ← 新增
```

### 修改 2: 上下文恢复（第 547-571 行）

```python
# Restore original test data
if original_test_data is not None:
    self.trainer.ctx.test_data = original_test_data
else:
    if 'test_data' in self.trainer.ctx:
        del self.trainer.ctx['test_data']

if original_test_loader is not None:
    self.trainer.ctx.test_loader = original_test_loader
else:
    if 'test_loader' in self.trainer.ctx:
        del self.trainer.ctx['test_loader']

if original_cur_split is not None:  # ← 新增
    self.trainer.ctx.cur_split = original_cur_split
else:
    if 'cur_split' in self.trainer.ctx:
        del self.trainer.ctx['cur_split']
```

### 修改 3: 错误处理中的恢复（第 492-521 行）

同样的恢复逻辑也应用于异常处理部分。

## 关键改进

1. **ReIterator 包装** - 确保 loader 格式与 trainer 期望一致
2. **设置 cur_split** - 确保 `check_split('test')` 能够通过
3. **完整的上下文恢复** - 包括 `cur_split` 的恢复或删除
4. **新增调试日志** - 显示 `check_split` 的结果

## 期望结果

重新运行训练后，应该看到：

### 调试日志
```
INFO: Domain amazon:
INFO:   - test_data set: True
INFO:   - test_loader set: True
INFO:   - test_data size: 100
INFO:   - test_loader batches: 25
INFO:   - cur_split: test
INFO:   - check_split result: True  ← 关键：必须是 True
```

### 不再有警告
```
# 应该不再看到这个警告
WARNING: No data for evaluation in split 'test'
```

### 准确率开始更新
```
Domain-specific Test Results:
  amazon      : Acc = 0.XXXX | Samples =  100
  webcam      : Acc = 0.XXXX | Samples =  100
  dslr        : Acc = 0.XXXX | Samples =   98
  caltech     : Acc = 0.XXXX | Samples =   90
```

**注意:** 初期准确率可能仍接近 0.1（随机猜测），但应该随训练逐渐提升。

## 验证步骤

### 1. 重新运行训练
```bash
python federatedscope/main.py --cfg scripts/example_configs/cross_domain_adaptive_office_caltech.yaml
```

### 2. 检查关键日志

在服务器端评估的日志中查找：
- ✓ `check_split result: True` - 必须是 True
- ✓ `test_loader batches: XX` - 必须 > 0
- ✗ 不应该有 "No data for evaluation" 警告

### 3. 观察准确率变化

- **第 1-5 轮:** 准确率可能在 0.0-0.15 之间（随机）
- **第 10 轮后:** 准确率应该开始提升
- **第 50 轮后:** 准确率应该明显高于随机水平

### 4. 如果仍然为 0

检查以下内容：
1. **`check_split result`** 是否为 True？
   - 如果是 False，数据设置有问题
   - 如果是 True，评估循环有其他问题

2. **客户端本地准确率** 是否提升？
   - 如果客户端准确率正常，只是服务器端为 0，可能是模型同步问题
   - 如果客户端也是 0，可能是模型或数据问题

3. **训练损失** 是否下降？
   - 如果损失不下降，模型可能没有学习
   - 检查学习率、优化器配置

## 为什么这个修复有效

1. **ReIterator 兼容性**
   - FederatedScope 的 trainer 期望使用 `ReIterator` 包装的 loader
   - `ReIterator` 提供了 `reset()` 方法，允许多次遍历
   - 直接使用 PyTorch DataLoader 可能导致迭代器耗尽

2. **cur_split 设置**
   - `check_split()` 方法检查 `{cur_split}_data` 和 `{cur_split}_loader`
   - 没有设置 `cur_split`，会导致检查失败
   - 评估会被跳过

3. **完整的上下文管理**
   - 保存和恢复所有相关的上下文变量
   - 避免影响后续的评估或训练
   - 确保多域评估能够顺利进行

## 修改总结

- **修改文件:** 1 个
  - `federatedscope/contrib/worker/cross_domain_adaptive.py`
- **修改位置:** 3 处
  - 第 453-481 行: 数据设置（添加 ReIterator 和 cur_split）
  - 第 492-521 行: 错误处理中的恢复
  - 第 547-571 行: 正常完成后的恢复
- **新增导入:** `from federatedscope.core.auxiliaries.ReIterator import ReIterator`
- **新增调试:** check_split 结果日志

## 后续观察

如果修复后：

1. **评估正常运行，准确率逐渐提升** → ✅ 问题解决
2. **评估正常运行，准确率停留在 ~10%** → 模型学习问题，检查超参数
3. **仍然有 "No data" 警告** → 检查 check_split 日志，可能需要进一步调试

**请重新运行并分享新的日志输出，特别是 `check_split result` 的值！**
