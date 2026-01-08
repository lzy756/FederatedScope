# GGEUR修复总结：解决低测试准确率问题

## 问题诊断

### 观察到的现象

从服务器训练日志 (`exp_print.log`) 分析：

1. **数据增强数量正确** ✅
   - Client 1: 380 → **1,250** samples (25 classes × 50)
   - Client 2: 1157 → **1,800** samples (36 classes × 50)
   - Client 3: 982 → **1,550** samples (31 classes × 50)
   - Client 4: 609 → **1,400** samples (28 classes × 50)

2. **训练准确率持续提升** ✅
   - Round 1: 12.9%
   - Round 10: 56.0%
   - 说明模型在学习

3. **测试准确率极低** ❌
   - Round 1: 1.23%
   - Round 10: 3.69%
   - **训练/测试差距：52.3%** （严重过拟合）

### 根本原因

从日志中发现协方差矩阵的特征值异常：

```
Server: Class 0 geometry:
  - Top 5 eigenvalues: [9.999975e-07, 9.999978e-07, ...]
```

**关键发现：所有特征值都接近 1e-6！**

**问题链条**：
1. `cov_regularization = 1e-6` 太大
2. 协方差矩阵被正则化项主导
3. 所有特征值被压制到 ~1e-6
4. 生成的样本方差太小，都集中在均值附近
5. **生成的样本缺乏多样性**
6. 模型在单一分布的合成数据上过拟合
7. 无法泛化到多样化的真实测试集

## 解决方案

已修改配置文件：`scripts/example_configs/ggeur_officehome_lds.yaml`

```yaml
ggeur:
  # CRITICAL FIX: 降低正则化，避免特征值被压制
  cov_regularization: 1e-08  # 从默认1e-06降低到1e-08
  eigenvalue_scaling: false  # 暂时关闭缩放
```

## 重新训练

```bash
python run_ggeur.py --cfg scripts/example_configs/ggeur_officehome_lds.yaml
```

## 预期效果

- 特征值：预计 1e-4 到 1e-2（真实方差）
- 测试准确率：预计 **20-30%**（显著提升）
- 过拟合程度：<20%（合理范围）
