# 域总数键错误修复

## 问题

```
ValueError: Results to be formatted should be include the dataset_num in the dict, with key = domain_total
```

## 原因

在加权平均结果中包含了非标准的键（`domain_results`, `total_samples`），Monitor 将这些键误认为是指标名，并期望相应的 `{metric}_total` 键。

## 修复

**文件:** `federatedscope/contrib/worker/cross_domain_adaptive.py`
**位置:** 第 582-605 行

### 修改前
```python
weighted_results = {
    'test_acc': weighted_acc,
    'test_total': total_samples,
    'domain_results': domain_results,      # ← 问题：非标准键
    'total_samples': total_samples         # ← 问题：非标准键
}
```

### 修改后
```python
# Calculate weighted average for all metrics
weighted_metrics = {}
for metric_name in metric_names:
    weighted_value = 0.0
    for domain in domain_results.keys():
        result = domain_results[domain]
        size = result['size']
        weight = size / total_samples if total_samples > 0 else 0
        metric_value = result['metrics'].get(metric_name, 0.0)
        weighted_value += metric_value * weight
    weighted_metrics[metric_name] = weighted_value

# Only include standard metric keys
weighted_results = {
    'test_total': total_samples,
}
# Add all weighted metrics (test_acc, test_loss, etc.)
weighted_results.update(weighted_metrics)
```

## 改进

1. **移除非标准键** - 不再包含 `domain_results` 和 `total_samples`
2. **自动计算所有指标** - 不仅计算 test_acc，还包括 test_loss 等所有配置的指标
3. **统一格式** - 确保结果格式与 Monitor 期望一致

## 期望输出

修复后，服务器端评估应该完整运行：

```
================================================================================
[Round X] Server-side Evaluation on Balanced Test Sets
================================================================================
Domain-specific Test Results:
--------------------------------------------------------------------------------
  amazon      : Acc = 0.XXXX | Samples =  100 | Weight = 0.258
  caltech     : Acc = 0.XXXX | Samples =   90 | Weight = 0.232
  dslr        : Acc = 0.XXXX | Samples =   98 | Weight = 0.253
  webcam      : Acc = 0.XXXX | Samples =  100 | Weight = 0.258
--------------------------------------------------------------------------------
  Weighted Avg: Acc = 0.XXXX | Total Samples = 388
================================================================================
```

## 关于准确率为 0

从你的日志看到所有域的准确率都是 0.0。这**可能是正常的**，特别是：

1. **训练初期（前 1-5 轮）**
   - 模型随机初始化
   - 10 类分类的随机准确率应该接近 10%
   - 0% 可能是因为批次小或运气不好

2. **检查客户端训练**
   - 查看客户端本地训练准确率是否在提升
   - 查看训练损失是否在下降
   - 如果客户端准确率正常，只是服务器评估为 0，继续观察几轮

3. **可能的问题**
   - 如果训练多轮后仍为 0，可能有模型或数据问题
   - 检查是否有其他错误日志

## 下一步

**重新运行训练：**
```bash
python federatedscope/main.py --cfg scripts/example_configs/cross_domain_adaptive_office_caltech.yaml
```

**观察：**
1. ✓ 不应再有 `domain_total` 错误
2. ✓ 服务器端评估应该完整运行
3. ? 准确率应该随训练逐渐提升

如果准确率持续为 0：
- 查看客户端本地准确率
- 查看训练损失
- 检查是否有其他错误
