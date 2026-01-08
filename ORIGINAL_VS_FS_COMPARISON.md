# 原始GGEUR实现 vs FederatedScope实现 - 完整对比

## 关键差异总结

### 1. 模型架构
**原始代码**:
```python
class MyNet(nn.Module):
    def __init__(self, num_classes=65):
        super(MyNet, self).__init__()
        self.fc3 = nn.Linear(512, num_classes)

    def forward(self, x):
        return F.softmax(self.fc3(x), dim=1)  # 返回概率！
```

**FederatedScope当前实现**:
```python
# 在ggeur_trainer.py中构建MLP
layers.append(nn.Linear(input_dim, num_classes))
# 输出logits，不是概率
```

**需要修改**: 添加softmax到forward输出

---

### 2. 训练参数
**原始代码**:
- 优化器: `Adam(lr=0.001)`
- 损失函数: `CrossEntropyLoss()`
- local_epochs: `1`
- batch_size: `16`
- communication_rounds: `50`

**FederatedScope当前实现**:
- 优化器: 配置文件中指定 (Adam, lr=0.001) ✓
- 损失函数: CrossEntropyLoss() ✓
- local_epochs: 配置文件中的 `local_update_steps: 1` ✓
- batch_size: 配置文件中的 `batch_size: 16` ✓

**状态**: 参数已匹配 ✓

---

### 3. 数据流
**原始代码**:
1. 离线运行 `prototype_cov_matrix_generate_features.py` 生成增强特征
2. 保存到文件: `argumented_clip_features/{domain}/client_{i}_class_{j}/final_embeddings_filled.npy`
3. FedAvg_GGEUR.py 加载这些预生成的特征
4. 直接训练

**FederatedScope当前实现**:
1. 在线提取CLIP特征
2. 在线生成增强数据
3. 训练

**需要修改**: 可以保持在线生成，但需要确保生成逻辑完全一致

---

### 4. 数据增强参数
**原始代码** (`prototype_cov_matrix_generate_features.py`):
- `num_per_sample`: 每个原始样本生成的数量（用于补全）
- `target_size=50`: 每个类别最终50个样本
- Step 1: 从原始样本扩充，随机选50个
- Step 2: 从其他domain prototypes生成，每个50个
- combine_samples: 合并确保target_size=50

**FederatedScope当前实现**:
- `n_samples_per_original: 50`
- `m_samples_per_prototype: 50`
- `step1_samples_per_class: 50`

**可能的问题**: 原始代码的combine_samples有特殊逻辑

---

### 5. 评估
**原始代码**:
- 每轮在4个domain的测试集上分别评估
- 计算平均准确率
- 测试数据从预先提取的CLIP features加载

**FederatedScope当前实现**:
- evaluate()函数重新提取CLIP features
- 可能有缓存

---

## 需要修改的优先级

### 🔴 P0 - 必须修改（可能导致2%准确率）

1. **模型forward输出softmax** - 必须匹配原始实现
2. **数据增强逻辑** - 确保combine_samples逻辑一致

### 🟡 P1 - 建议修改（可能影响性能）

3. **测试数据加载** - 使用预提取的CLIP features而非每次重新提取

### 🟢 P2 - 优化（不影响准确率）

4. **日志和监控** - 添加更多调试信息

---

## 修改方案

见后续文件...
