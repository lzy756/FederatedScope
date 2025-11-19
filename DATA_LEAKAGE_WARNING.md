# 数据泄漏问题说明和解决方案

## ⚠️ 当前存在的数据泄漏问题

### 问题描述

当前实现中，服务器端的平衡测试集是从**整个数据集**中随机采样的，这意味着：

1. **服务器测试样本** 可能与 **客户端训练样本** 重叠
2. **服务器测试样本** 可能与 **客户端测试样本** 重叠
3. 导致评估结果**过于乐观**，不能准确反映模型的泛化能力

### 具体影响

```
整个 Office-Caltech 数据集
├── 客户端数据 (通过 dirichlet split 分配给20个客户端)
│   ├── 训练集 (80%)  ← 可能包含服务器测试样本！
│   ├── 验证集 (10%)  ← 可能包含服务器测试样本！
│   └── 测试集 (10%)  ← 可能包含服务器测试样本！
└── 服务器测试集 (200 samples/domain) ← 从整个数据集采样
```

## ✅ 解决方案

### 方案 1：接受数据泄漏（仅用于研究/对比实验）

**适用场景：**
- 只是为了对比不同方法在相同设置下的性能
- 所有对比方法都使用相同的数据分割方式
- 明确标注结果可能存在数据泄漏

**当前实现：**
```python
# 从整个数据集采样
server_test_samples_per_class: 20  # 200 samples/domain
```

**优点：**
- 实现简单
- 所有域都能获得足够的测试样本
- 样本分布均衡

**缺点：**
- ⚠️ 数据泄漏
- ⚠️ 评估结果过于乐观
- ⚠️ 不能真实反映泛化能力

### 方案 2：使用独立的held-out测试集（推荐）

**最佳实践：**

1. **数据预处理阶段**：将数据集分为三部分
   ```
   整个 Office-Caltech 数据集
   ├── 训练池 (90%)
   │   └── 分配给客户端，再分为 train/val/test
   └── Held-out 测试集 (10%)
       └── 仅用于最终评估，不参与训练
   ```

2. **实现步骤**：

   a. 修改数据加载逻辑，先预留10%作为held-out set

   b. 客户端只使用剩余90%的数据

   c. 服务器测试集从held-out set中采样

**修改建议：**

```python
# 在 OfficeCaltech 类中添加 held_out 选项
class OfficeCaltech(Dataset):
    def __init__(self,
                 root,
                 domain='amazon',
                 split='train',  # 'train', 'val', 'test', 'held_out'
                 use_held_out=False,  # 是否预留held-out set
                 held_out_ratio=0.1,  # held-out set比例
                 ...):
        # 先划分 held-out
        # 再从剩余数据划分 train/val/test
```

### 方案 3：只从每个域的test split采样（部分解决）

**实现：**
```python
def load_balanced_office_caltech_data(...):
    # 只从test split (10%) 采样
    full_test_dataset = OfficeCaltech(
        root=root,
        domain=domain,
        split='test',  # 只用test split
        ...
    )
    # 从test split中采样
```

**优点：**
- 相对简单
- 避免了与训练集的直接重叠

**缺点：**
- 小域（dslr, webcam）样本仍然不足
- 仍然可能与客户端的test set重叠
- 测试集大小不均衡

## 📊 当前状态

当前实现使用**方案1**，因为：
1. 简单易实现
2. 确保所有域有足够样本
3. 作为baseline对比实验可以接受

但请注意：
- ⚠️ 结果会包含数据泄漏
- ⚠️ 不应该用于声称真实泛化性能
- ⚠️ 如果要发表论文，建议使用方案2

## 🔧 如何切换到方案2

如果你需要更严格的评估，可以：

1. 修改数据集类，添加held-out split支持
2. 修改splitter逻辑，先预留held-out set
3. 修改服务器测试集加载，使用held-out set

具体实现可以参考FederatedScope中其他数据集的held-out实现。

## 📝 建议

**对于当前实验：**
- 如果只是内部对比不同FL方法，当前实现可以接受
- 在报告结果时，注明"使用相同的测试集分割"
- 如果要声称泛化性能，必须使用held-out set

**对于论文发表：**
- 强烈建议实现方案2
- 明确说明数据分割策略
- 避免数据泄漏的质疑
