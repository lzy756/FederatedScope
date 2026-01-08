# GGEUR 低准确率问题 - 完整诊断指南

## 当前状态
- 准确率：0.82%, 1.23%, 1.64% （接近随机猜测 1.54%）
- Loss: ~4.17 （等于 -log(1/65)，即随机猜测的loss）
- 配置：已恢复 lda_domain splitter，local_update_steps=1，use_softmax_output=false

## 关键发现

### 1. 原始实现 vs FederatedScope 实现的差异

| 方面 | 原始实现 | FederatedScope 实现 |
|------|----------|---------------------|
| 训练数据 | 从 .npy 文件加载（预生成） | 动态生成（运行时增强） |
| 测试数据 | 从 .npy 文件加载 | 动态提取 CLIP 特征 |
| CLIP 模型 | 可能是 open_clip | 使用 openai/CLIP |
| 数据增强 | 离线预处理 | 在线实时生成 |

### 2. 可能的问题原因

#### A. CLIP 模型版本不匹配 ⚠️ **最可能**
- 原始实现可能使用 open_clip（不同的权重）
- FederatedScope 使用 openai/CLIP
- **解决方案**: 检查原始代码的 CLIP 导入，确保使用相同的模型

#### B. 数据增强质量问题
- 增强算法虽然参数匹配，但生成的数据质量可能不同
- **检查**: 增强后每个客户端是否有 3250 个样本（65类×50）

#### C. 训练/测试数据不一致
- 训练时用增强的嵌入，测试时重新提取，可能不匹配
- **检查**: CLIP 特征提取是否完全一致（包括预处理）

#### D. 优化器状态问题
- 原始实现每轮重新创建优化器
- FederatedScope 可能重用优化器，导致动量累积
- **检查**: ggeur_trainer.py 的优化器初始化

#### E. 数据分布问题
- lda_domain 可能导致某些客户端缺少关键类别
- 数据增强应该补偿，但如果增强失败，会导致问题
- **检查**: 每个客户端是否有所有 65 个类的数据

## 诊断步骤

### 步骤 1: 运行诊断脚本
```bash
python diagnose_ggeur_detailed.py
```

检查输出，确认：
- ✓ CLIP 缓存文件存在
- ✓ 日志中有 "DATA DISTRIBUTION REPORT"
- ✓ 每个客户端增强后有 ~3250 个样本
- ✓ 所有 65 个类都存在

### 步骤 2: 检查训练日志

查找关键信息：
```
CLIENT #1 - DATA DISTRIBUTION REPORT
  Original data: XXX samples
  Augmented data: 3250 samples  ← 应该在这个数量级

  Per-class distribution:
    Class 0: 50 samples  ← 每个类应该有 50 个样本
    Class 1: 50 samples
    ...
    Class 64: 50 samples
```

### 步骤 3: 检查第1轮训练
```
Round 1:
  Training Loss: X.XXXX
  Training Acc: 0.XXXX  ← 应该 > 0.05 (5%)

  Test Acc: 0.XXXX  ← 应该 > 0.15 (15%)
```

**如果第1轮 train_acc < 5%**：数据或模型有问题
**如果第1轮 train_acc > 50% 但 test_acc < 5%**：训练/测试数据不匹配

### 步骤 4: 对比原始源码

运行原始源码并记录：
1. 最终准确率（每个域）
2. 第1轮的准确率
3. Loss 的下降曲线
4. 使用的 CLIP 模型（检查导入语句）

## 快速测试

创建一个简化测试来验证核心功能：

```python
# test_ggeur_core.py
import torch
from federatedscope.contrib.data_augmentation.ggeur_augmentation import GGEURAugmentation

# 模拟全局几何形状
global_geometries = {}
for class_id in range(65):
    eigenvalues = torch.linspace(10, 0.1, 50)
    eigenvectors = torch.randn(512, 50)
    eigenvectors, _ = torch.linalg.qr(eigenvectors)
    global_geometries[class_id] = {
        'eigenvalues': eigenvalues,
        'eigenvectors': eigenvectors
    }

# 创建增强器
augmenter = GGEURAugmentation(
    global_geometries=global_geometries,
    scenario='multi_domain',
    device='cpu'
)

# 测试增强
embeddings = torch.randn(100, 512)  # 100 个样本
embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
labels = torch.randint(0, 65, (100,))

# 模拟其他域的原型
other_prototypes = {}
for class_id in range(65):
    other_prototypes[class_id] = {
        1: torch.randn(512),
        2: torch.randn(512),
        3: torch.randn(512)
    }

# 增强
aug_emb, aug_lab = augmenter.augment_dataset(
    embeddings, labels,
    other_domain_prototypes=other_prototypes,
    N=50, M=50, target_size=50
)

print(f"原始样本: {len(embeddings)}")
print(f"增强后样本: {len(aug_emb)}")
print(f"预期: 65 × 50 = 3250")
print(f"每个类的样本数: {[((aug_lab == c).sum().item()) for c in range(min(10, 65))]}")
```

## 最可能的修复方案

### 修复 #1: 确保使用正确的 CLIP 模型

检查原始代码中的 CLIP 导入：
```bash
grep -n "import clip\|from clip\|open_clip" "ProjectsFederatedScopeGGEUR_source/Multi Domain/Office-Home-LDS/*.py"
```

如果使用的是 open_clip，需要修改 CLIPExtractor：
```python
# federatedscope/contrib/utils/clip_extractor.py
# 替换 line 52:
import open_clip  # 而不是 import clip
model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-16', pretrained='laion2b_s34b_b88k')
```

### 修复 #2: 增加 local_update_steps

虽然原始实现用 1，但可能需要更多训练：
```yaml
train:
  local_update_steps: 10  # 尝试增加到 10
```

### 修复 #3: 检查优化器重置

在 ggeur_trainer.py 的 `_hook_on_fit_start_init` 中，确保每轮重新创建优化器。

## 下一步

1. **立即执行**: 运行 `diagnose_ggeur_detailed.py` 并提供输出
2. **关键问题**:
   - 原始源码使用的是 `import clip` 还是 `import open_clip`？
   - 原始源码的最终准确率是多少？
3. **对比测试**:
   - 用原始源码运行，记录详细日志
   - 对比数据分布、准确率趋势

一旦确认了 CLIP 模型版本，问题应该就能解决！
