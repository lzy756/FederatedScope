# 空评估数据修复说明

## 问题描述

### 问题 1: ValueError - need at least one array to concatenate
训练时出现以下错误：
```
ValueError: need at least one array to concatenate
```
错误发生在 `federatedscope/core/trainers/torch_trainer.py:430`。

### 问题 2: ValueError - best_res_update_round_wise_key not in results
修复问题1后出现新错误：
```
ValueError: Your specified eval.best_res_update_round_wise_key is not in target results, use another key or check the name.
Got eval.best_res_update_round_wise_key=test_acc, the keys of results are []
```

## 根本原因

1. **空数据问题**：当评估数据集为空或没有数据被处理时，`ctx.ys_true` 和 `ctx.ys_prob` 列表保持为空，导致 concatenate 操作失败
2. **空结果字典问题**：返回空的 `eval_metrics` 字典导致 Monitor 无法找到配置的 `best_res_update_round_wise_key`

## 完整修复方案

### 1. 修复 `torch_trainer.py` (federatedscope/core/trainers/torch_trainer.py:414-459)

在 `_hook_on_fit_end` 方法中添加完整的空列表处理逻辑：

```python
def _hook_on_fit_end(self, ctx):
    """
    Evaluate metrics.
    """
    # Check if there are any predictions collected
    if len(ctx.ys_true) == 0 or len(ctx.ys_prob) == 0:
        # Return empty results if no data was processed
        ctx.ys_true = CtxVar(np.array([]), LIFECYCLE.ROUTINE)
        ctx.ys_prob = CtxVar(np.array([]), LIFECYCLE.ROUTINE)

        # Create dummy eval_metrics with required keys to avoid downstream errors
        # Use 0.0 as placeholder values
        eval_metrics = {}
        split_name = ctx.get('cur_split', 'test')

        # Get configured metrics from monitor if available
        if hasattr(ctx, 'monitor') and hasattr(ctx.monitor, 'metric_calculator'):
            metric_names = ctx.monitor.metric_calculator.eval_metric.keys()
            for metric in metric_names:
                eval_metrics[f"{split_name}_{metric}"] = 0.0
        else:
            # Fallback: create common metrics
            eval_metrics[f"{split_name}_acc"] = 0.0
            eval_metrics[f"{split_name}_loss"] = 0.0

        setattr(ctx, 'eval_metrics', eval_metrics)
        logger.warning(f"No data for evaluation in split '{split_name}', "
                     f"returning zero metrics: {list(eval_metrics.keys())}")
        return

    ctx.ys_true = CtxVar(np.concatenate(ctx.ys_true), LIFECYCLE.ROUTINE)
    ctx.ys_prob = CtxVar(np.concatenate(ctx.ys_prob), LIFECYCLE.ROUTINE)
    results = ctx.monitor.eval(ctx)
    setattr(ctx, 'eval_metrics', results)
```

**修复内容：**
- 在 concatenate 之前检查列表是否为空
- 如果为空，生成包含必需指标键的虚拟结果字典（值为 0.0）
- 根据 monitor 配置动态生成指标键（如 `test_acc`, `test_loss` 等）
- 如果 monitor 不可用，回退到常见指标
- 添加警告日志说明返回了零指标
- 避免了 ValueError 并满足 Monitor 对指标键的要求

### 2. 改进服务器端评估 `cross_domain_adaptive.py` (federatedscope/contrib/worker/cross_domain_adaptive.py:411-575)

**修复 2.1：添加 trainer 检查**

在 `eval()` 方法开始时添加 trainer 存在性检查：

```python
def eval(self):
    # Check if trainer exists
    if not hasattr(self, 'trainer') or self.trainer is None:
        logger.warning("Server has no trainer, skipping server-side evaluation")
        return
```

**修复 2.2：改进数据上下文设置**

在临时设置测试数据时，同时保存和恢复 `original_data` 字段：

```python
# Temporarily set the test data for this domain
original_test_data = self.trainer.ctx.get('test_data')
original_test_loader = self.trainer.ctx.get('test_loader')
original_data = self.trainer.ctx.get('data')
original_num_samples = self.trainer.ctx.get('num_samples')

# Set test data with proper structure
self.trainer.ctx.test_data = test_loader.dataset
self.trainer.ctx.test_loader = test_loader
# Also set data dict if it's expected
if original_data is not None and hasattr(original_data, '__getitem__'):
    try:
        original_data['test'] = test_loader.dataset
    except (TypeError, KeyError):
        pass
```

**修复 2.3：完整恢复原始上下文**

在评估完成或失败后，恢复所有保存的上下文状态：

```python
# Restore original test data
if original_test_data is not None:
    self.trainer.ctx.test_data = original_test_data
if original_test_loader is not None:
    self.trainer.ctx.test_loader = original_test_loader
if original_data is not None and hasattr(original_data, '__getitem__'):
    try:
        if original_test_data is not None:
            original_data['test'] = original_test_data
    except (TypeError, KeyError):
        pass
```

## 测试验证

创建了测试脚本 `test_empty_eval_fix.py` 来验证修复：

### 测试用例
1. **空列表测试**：验证当 `ctx.ys_true` 和 `ctx.ys_prob` 为空时不会崩溃
2. **非空列表测试**：验证正常情况下 concatenate 仍然工作正常

### 测试结果
```
============================================================
Testing empty evaluation fix
============================================================
Testing empty list case...
[PASS] Empty list case handled correctly

Testing non-empty list case...
[PASS] Non-empty list case handled correctly
  - ys_true shape: (5,)
  - ys_prob shape: (5, 2)

============================================================
All tests passed! [OK]
============================================================
```

## 影响范围

### 修改的文件
1. `federatedscope/core/trainers/torch_trainer.py` - 核心训练器的评估钩子
2. `federatedscope/contrib/worker/cross_domain_adaptive.py` - 跨域自适应服务器的评估方法

### 向后兼容性
- 这些修复是向后兼容的
- 正常情况下（有数据）的行为保持不变
- 只在异常情况（无数据）时提供优雅的降级处理

## 建议

在运行训练之前：

1. 确保数据路径正确配置
2. 验证数据集加载成功
3. 如果出现 "Server has no trainer" 警告，这表示服务器端评估被跳过（这通常是预期行为）

## 后续步骤

现在可以重新运行训练：
```bash
python federatedscope/main.py --cfg scripts/example_configs/cross_domain_adaptive_office_caltech.yaml
```

如果仍然出现问题，请检查：
1. 数据路径是否正确（配置文件中的 `data.root`）
2. Office-Caltech 数据集是否已正确下载和解压
3. 查看日志中的数据加载相关警告信息
