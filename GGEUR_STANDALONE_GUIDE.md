# GGEUR Standalone Implementation - 使用指南

## 🎯 目标

创建一个完全模拟原始FedAvg_GGEUR.py的实现，绕过FederatedScope的复杂架构，直接使用离线预提取的.npy特征文件。

## 📁 文件说明

### 1. `run_ggeur_standalone.py`
- **功能**: 主训练脚本，完全模拟原始FedAvg_GGEUR.py
- **输入**: 预提取的.npy特征文件
- **输出**: 训练好的模型和准确率报告

### 2. `prepare_ggeur_data.py`
- **功能**: 离线数据准备脚本
- **操作**:
  - 提取训练集CLIP特征
  - 提取测试集CLIP特征
  - 计算全局协方差矩阵
  - (增强特征需要使用原始代码生成)

## 🚀 使用步骤

### Step 1: 提取CLIP特征和计算协方差矩阵

```bash
python prepare_ggeur_data.py
```

这会生成：
- `clip_office_home_train_features/` - 训练集特征
- `clip_office_home_test_features/` - 测试集特征
- `cov_matrix_output/` - 协方差矩阵

### Step 2: 生成增强特征（使用原始代码）

由于数据增强逻辑复杂，建议直接使用原始代码生成：

```bash
cd "D:\Projects\2025CVPR_GGEUR\Multi Domain\Office-Home-LDS"

# 如果还没有生成过，先提取特征
python train_client_class_clip_features2tensor.py
python test_clip_features2tensor.py
python prototype_clip_features2tensor.py
python clip_tensor2aggregate_covariance_matrix.py

# 生成增强特征
python prototype_cov_matrix_generate_features.py
```

然后复制生成的目录：

```bash
# 在PowerShell或CMD中
cp -r "D:\Projects\2025CVPR_GGEUR\Multi Domain\Office-Home-LDS\argumented_clip_features" "D:\Projects\FederatedScope\"

# 或者直接在文件管理器中复制
```

### Step 3: 运行训练

```bash
cd "D:\Projects\FederatedScope"
python run_ggeur_standalone.py
```

## 📊 预期结果

如果实现正确，应该能看到：

```
Round 1/50, Art Accuracy: 0.xxxx, Clipart Accuracy: 0.xxxx, Product Accuracy: 0.xxxx, Real_world Accuracy: 0.xxxx, Average Accuracy: 0.xxxx
Round 2/50, Art Accuracy: 0.xxxx, Clipart Accuracy: 0.xxxx, Product Accuracy: 0.xxxx, Real_world Accuracy: 0.xxxx, Average Accuracy: 0.xxxx
...
Round 50/50, Art Accuracy: 0.70+, Clipart Accuracy: 0.70+, Product Accuracy: 0.70+, Real_world Accuracy: 0.70+, Average Accuracy: 0.70+
```

**期望准确率**: ~70% (与原始论文一致)

## 🔍 与原始代码的对比

| 方面 | 原始代码 | Standalone实现 | 状态 |
|------|---------|----------------|------|
| 模型架构 | MyNet with softmax | 完全相同 | ✅ |
| 数据加载 | 加载.npy文件 | 完全相同 | ✅ |
| 训练循环 | FedAvg | 完全相同 | ✅ |
| local_epochs | 1 | 1 | ✅ |
| 测试流程 | 加载预提取特征 | 完全相同 | ✅ |
| 输出格式 | Round x/50, ... | 完全相同 | ✅ |

## ⚠️ 注意事项

1. **路径要求**
   - 所有.npy文件必须在正确的位置
   - 文件命名必须完全匹配原始格式

2. **数据格式**
   - CLIP特征: float32, shape=(N, 512)
   - 标签: int64, shape=(N,)
   - 增强特征每类50个样本

3. **设备要求**
   - GPU可用时自动使用CUDA
   - 无GPU时使用CPU (会很慢)

## 🐛 故障排除

### 问题1: 找不到特征文件
```
错误: 增强特征目录不存在: ./argumented_clip_features
```
**解决**: 运行Step 2，从原始代码生成增强特征

### 问题2: 准确率仍然很低 (~2%)
**可能原因**:
1. 增强特征生成不正确
2. CLIP模型版本不匹配
3. 数据归一化问题

**检查方法**:
```python
# 检查增强特征的norm
import numpy as np
features = np.load('argumented_clip_features/Art/client_0_class_0/final_embeddings_filled.npy')
norms = np.linalg.norm(features, axis=1)
print(f"Mean norm: {norms.mean():.4f}")  # 应该接近1.0
```

### 问题3: 客户端数量不对
```
错误: 应该有4个客户端，实际加载了X个
```
**解决**: 检查每个domain的client_id是否匹配：
- Art → client_0
- Clipart → client_1
- Product → client_2
- Real_World → client_3

## 📈 性能对比

| 方法 | 准确率 | 特点 |
|------|--------|------|
| FedAvg (原始) | ~50% | 无数据增强 |
| GGEUR (原始) | ~70% | 离线数据增强 |
| FederatedScope (修复前) | ~2% | 在线特征提取 ❌ |
| Standalone实现 | ~70% (预期) | 离线特征 ✅ |

## 🎓 关键经验教训

1. **离线 > 在线**: CLIP特征提取必须离线完成
2. **一致性至关重要**: 训练和测试必须使用相同的特征提取过程
3. **简单就是好**: 不需要复杂的架构，简单的FedAvg循环就够了
4. **完全复现**: 与其修改原始代码，不如完全复现

## 下一步

如果standalone实现成功（准确率~70%），可以考虑：
1. 将成功的逻辑集成回FederatedScope
2. 修改FederatedScope的数据加载流程
3. 添加离线特征提取选项

**但现在最重要的是：先让standalone版本工作！**
