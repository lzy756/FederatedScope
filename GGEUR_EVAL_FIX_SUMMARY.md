# GGEUR 评估错误修复总结

## 日期：2025-12-19

## 问题历史

### 错误 1：`ValueError: eval.best_res_update_round_wise_key is not in target results`
**原因：** 配置中设置为 `val_loss`，但实际结果只有 `['acc', 'correct', 'loss', 'total', 'avg_loss']`

**修复：**
- 将所有配置文件的 `best_res_update_round_wise_key` 改为 `'acc'`
- 添加 `split: ['test']` 明确评估数据集

### 错误 2：`UnboundLocalError: local variable 'update_key' referenced before assignment`
**原因：** `monitor.py` 中的代码假设 key 必须有前缀，但实际 `'acc'` 没有前缀

**修复：**
- 在 `monitor.py` 中初始化 `update_key = self.round_wise_update_key`
- 添加 `break` 语句避免覆盖
- 改进错误消息显示可用指标

### 错误 3：`ValueError: Results should include dataset_num with key = acc_total`
**原因：** GGEUR trainer 返回的指标缺少数据集前缀（如 `test_`）

**修复：**
- 修改 `ggeur_trainer.py` 的 `evaluate()` 方法
- 返回带前缀的键：`test_acc`, `test_total`, `test_correct` 等

### 错误 4：`KeyError: 'acc'`
**原因：** `best_res_update_round_wise_key` 配置为 `'acc'`，但实际键是 `'test_acc'`

**修复：**
- 更新所有配置文件：`best_res_update_round_wise_key: 'test_acc'`
- 添加 `split: ['test']` 确保评估正确的数据集

## 最终修复

### 1. Monitor (monitor.py:64-78)
```python
# Obtain the whether the larger the better
self.round_wise_update_key = cfg.eval.best_res_update_round_wise_key

# Extract the metric name without mode prefix
update_key = self.round_wise_update_key  # Default to the full key
for mode in ['train', 'val', 'test']:
    if mode in self.round_wise_update_key:
        update_key = self.round_wise_update_key.split(f'{mode}_')[1]
        break

assert update_key in self.metric_calculator.eval_metric, \
    f'{update_key} not found in metrics. ' \
    f'Available metrics: {list(self.metric_calculator.eval_metric.keys())}'
```

### 2. GGEUR Trainer (ggeur_trainer.py:289-307)
```python
# Return metrics in the standard format expected by the framework
# Need to add the split name prefix (e.g., 'test_acc', 'test_total')
metrics = {
    f'{target_data_split_name}_acc': accuracy,
    f'{target_data_split_name}_correct': correct,
    f'{target_data_split_name}_loss': loss,
    f'{target_data_split_name}_total': len(test_labels),
    f'{target_data_split_name}_avg_loss': loss,
}

# Save for client's formatted output
self._eval_results = metrics

return metrics
```

### 3. 所有配置文件
```yaml
eval:
  freq: 5
  metrics: ['acc']
  best_res_update_round_wise_key: 'test_acc'  # 必须使用带前缀的键名
  split: ['test']  # 明确指定评估数据集
```

## 修改的文件列表

### 核心代码
1. ✅ `federatedscope/core/monitors/monitor.py`
   - 修复 `update_key` 初始化问题
   - 添加 `break` 语句
   - 改进错误消息

2. ✅ `federatedscope/contrib/trainer/ggeur_trainer.py`
   - 评估指标添加数据集前缀
   - 保存评估结果到 `self._eval_results`

3. ✅ `federatedscope/contrib/worker/ggeur_client.py`
   - 更新 `callback_funcs_for_evaluate()`
   - 更新 `_print_eval_results()` 使用新的键名格式

### 配置文件
4. ✅ `scripts/example_configs/ggeur_pacs_multi_domain.yaml`
5. ✅ `scripts/example_configs/ggeur_digits_multi_domain.yaml`
6. ✅ `scripts/example_configs/ggeur_officehome_lds.yaml`

所有配置文件都更新为：
- `best_res_update_round_wise_key: 'test_acc'`
- `split: ['test']`

## 指标格式变化

### 之前（错误）
```python
{
    'acc': 0.85,
    'total': 1000,
    'correct': 850,
    'loss': 0.42,
    'avg_loss': 0.42
}
```

### 现在（正确）
```python
{
    'test_acc': 0.85,
    'test_total': 1000,
    'test_correct': 850,
    'test_loss': 0.42,
    'test_avg_loss': 0.42
}
```

## 配置说明

### 关键配置项

#### `best_res_update_round_wise_key`
- **格式：** `{split}_{metric}`
- **示例：** `'test_acc'`, `'val_loss'`, `'train_acc'`
- **规则：** 必须与评估返回的指标键匹配

#### `split`
- **格式：** 列表，如 `['test']` 或 `['test', 'val']`
- **说明：** 指定要评估的数据集分割
- **默认：** 如果不指定，可能使用默认值

### 正确的配置示例

```yaml
eval:
  freq: 5
  metrics: ['acc']
  best_res_update_round_wise_key: 'test_acc'
  split: ['test']

early_stop:
  patience: 50
  delta: 0.0
  improve_indicator_mode: 'best'
```

## 验证方法

### 1. 检查指标键名
运行时观察日志输出：
```
Evaluation on test: Acc=0.8524, Loss=0.4231
```

### 2. 检查 history_results
在 `client.py` 中，`self.history_results` 应该包含：
- `test_acc`
- `test_total`
- `test_correct`
- `test_loss`

### 3. 运行验证脚本
```bash
python verify_ggeur_config.py
```

应该看到：
```
Trainer metrics format:
  test_acc: 0.85
  test_correct: 850
  test_loss: 0.42
  test_total: 1000
  test_avg_loss: 0.42

All verifications passed!
```

## 运行命令

现在可以正常运行：

```bash
# Office-Home
python run_ggeur.py --cfg scripts/example_configs/ggeur_officehome_lds.yaml

# PACS
python run_ggeur.py --cfg scripts/example_configs/ggeur_pacs_multi_domain.yaml

# Digits
python run_ggeur.py --cfg scripts/example_configs/ggeur_digits_multi_domain.yaml
```

## 预期输出

### 数据分布报告
```
================================================================================
CLIENT #1 - DATA DISTRIBUTION REPORT
================================================================================

Original data: 520 samples
Augmented data: 3250 samples
Augmentation ratio: 6.25x

Per-class distribution:
Class    Original     Augmented    Proportion   Ratio
--------------------------------------------------------------------------------
0        8            50           0.0154       6.25x
...
================================================================================
```

### 评估结果（如果实现了格式化输出）
```
================================================================================
ROUND 1 - CLIENT #1 - EVALUATION RESULTS
================================================================================
TEST     Accuracy: 0.8524 (85.24%)
================================================================================
```

## 注意事项

### 1. 键名一致性
确保以下三者匹配：
- Trainer 返回的键名：`test_acc`
- 配置的 `best_res_update_round_wise_key`：`test_acc`
- Monitor 提取的键名：`acc`（自动从 `test_acc` 提取）

### 2. 数据集分割
- 使用 `split: ['test']` 明确指定评估数据集
- 如果需要多个分割：`split: ['test', 'val']`

### 3. 向后兼容性
修复后的代码兼容两种格式：
- 带前缀：`test_acc`, `val_loss`
- 不带前缀：`acc`, `loss`（会使用原始键名）

## 相关文档

1. **目标分布功能**：`GGEUR_TARGET_DISTRIBUTION_GUIDE.md`
2. **新功能总结**：`GGEUR_NEW_FEATURES_SUMMARY.md`
3. **修复总结**：`GGEUR_FIX_SUMMARY.md`
4. **差异分析**：`GGEUR_IMPLEMENTATION_DIFFERENCES.md`

## 总结

所有评估相关的错误已修复：
- ✅ Monitor 初始化错误
- ✅ 指标键名格式错误
- ✅ 配置键名不匹配
- ✅ 所有配置文件已更新

系统现在应该可以正常运行！
