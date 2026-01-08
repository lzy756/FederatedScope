# GGEUR 准确率问题诊断和修复

## 发现的关键问题

通过对比源码 `D:\Projects\2025CVPR_GGEUR\` 发现了导致准确率低的**3个关键差异**：

### 问题 1：特征值缩放顺序错误 ❌ **严重**

**源码实现**（正确的，虽然看起来奇怪）：
```python
# D:\Projects\2025CVPR_GGEUR\Multi Domain\Office-Home-LDS\prototype_cov_matrix_generate_features.py
def nearest_pos_def(cov_matrix):
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)  # 返回升序：[小 -> 大]
    scale_factors = np.ones_like(eigenvalues)
    scale_factors[:10] = np.linspace(5, 1, 10)  # 缩放最小的10个！
    eigenvalues = eigenvalues * scale_factors
    eigenvalues[eigenvalues < 0] = 0
    return eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
```

**我们的实现**（错误）：
```python
# federatedscope/contrib/utils/geometry_utils.py
def eigendecompose(...):
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance_matrix)  # 升序
    # 然后排序成降序！
    sorted_indices = torch.argsort(eigenvalues, descending=True)
    eigenvalues = eigenvalues[sorted_indices]  # [大 -> 小]
    eigenvectors = eigenvectors[:, sorted_indices]
    # ...

def _apply_eigenvalue_scaling(eigenvalues):
    scale_factors[:n_scale] = torch.linspace(5.0, 1.0, n_scale)
    # 实际缩放的是最大的10个！❌
```

**问题解释**：
- 源码：`np.linalg.eigh` 返回升序特征值，缩放 `[:10]` = 缩放最小的10个
- 我们：先降序排序，然后缩放 `[:10]` = 缩放最大的10个
- **这完全改变了算法的行为！**

### 问题 2：模型输出层定义不同 ❌ **重要**

**源码实现**：
```python
# D:\Projects\2025CVPR_GGEUR\Multi Domain\Office-Home-LDS\FedAvg_GGEUR.py
class MyNet(nn.Module):
    def __init__(self, num_classes=65):
        super(MyNet, self).__init__()
        self.fc3 = nn.Linear(512, num_classes)

    def forward(self, x):
        return F.softmax(self.fc3(x), dim=1)  # 返回概率！

# 训练时
criterion = nn.CrossEntropyLoss()  # 内部会再做一次 log_softmax
```

**我们的实现**：
```python
# federatedscope/contrib/trainer/ggeur_trainer.py
class MLP:
    def forward(self, x):
        return self.fc(x)  # 返回 logits

# 训练时
criterion = nn.CrossEntropyLoss()  # 这是正确的用法
```

**问题解释**：
- 源码：forward 返回 softmax 结果，然后 CrossEntropyLoss 内部再做 log_softmax（虽然这在技术上是错误的）
- 我们：forward 返回 logits，CrossEntropyLoss 做 log_softmax（这是标准用法）
- **虽然源码的用法不标准，但我们必须复现它才能得到相同结果**

### 问题 3：协方差矩阵归一化 ⚠️ **可能**

**源码**（第60行）：
```python
# 除以 n_samples（有偏估计）
covariance = torch.mm(centered.t(), centered) / n_samples
```

**我们**（geometry_utils.py:60）：
```python
# 除以 n_samples（同样是有偏估计）
covariance = torch.mm(centered.t(), centered) / n_samples
```

这个是一致的，不是问题。

---

## 修复方案

### 修复 1：特征值缩放顺序

需要修改 `federatedscope/contrib/utils/geometry_utils.py`：

#### 方案A：不排序，直接用升序（与源码完全一致）

```python
def eigendecompose(covariance_matrix: torch.Tensor,
                  top_k: Optional[int] = None,
                  min_eigenvalue: float = 1e-8) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Perform eigenvalue decomposition on covariance matrix.

    NOTE: For GGEUR_Clip, we keep eigenvalues in ASCENDING order (as returned by eigh)
    to match the paper's implementation which scales the SMALLEST 10 eigenvalues.
    """
    # Eigenvalue decomposition (returns ASCENDING order)
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance_matrix)

    # DO NOT SORT! Keep ascending order to match paper's nearest_pos_def
    # scale_factors[:10] will scale the SMALLEST 10 eigenvalues

    # Filter out very small eigenvalues
    valid_mask = eigenvalues > min_eigenvalue
    eigenvalues = eigenvalues[valid_mask]
    eigenvectors = eigenvectors[:, valid_mask]

    if top_k is not None and top_k < len(eigenvalues):
        # Take LAST k (largest) for top_k
        eigenvalues = eigenvalues[-top_k:]
        eigenvectors = eigenvectors[:, -top_k:]

    return eigenvalues, eigenvectors
```

#### 方案B：继续降序，但修改缩放逻辑

```python
def _apply_eigenvalue_scaling(eigenvalues: torch.Tensor) -> torch.Tensor:
    """
    Apply eigenvalue scaling as in paper's nearest_pos_def function.

    NOTE: Paper scales the SMALLEST 10 eigenvalues (because eigh returns ascending).
    Since we sort to descending, we need to scale the LAST 10.
    """
    eigenvalues = eigenvalues.clone()
    scale_factors = torch.ones_like(eigenvalues)

    # Scale SMALLEST 10 eigenvalues (at the end after descending sort)
    n_scale = min(10, len(eigenvalues))
    scale_factors[-n_scale:] = torch.linspace(5.0, 1.0, n_scale,
                                              dtype=eigenvalues.dtype,
                                              device=eigenvalues.device)

    scaled_eigenvalues = eigenvalues * scale_factors
    return torch.clamp(scaled_eigenvalues, min=0)
```

**推荐方案A**，因为更简单且与源码完全一致。

### 修复 2：模型输出层

需要修改 `federatedscope/contrib/trainer/ggeur_trainer.py`：

```python
class GGEURTrainer(GeneralTorchTrainer):

    def _build_mlp_classifier(self):
        """Build MLP classifier for embedding space."""
        cfg = self._cfg.ggeur

        input_dim = cfg.embedding_dim
        num_classes = self._cfg.model.num_classes
        num_layers = cfg.get('mlp_layers', 1)

        if num_layers == 1:
            # Single layer MLP (linear classifier)
            # Wrap with softmax to match paper's implementation
            self.mlp_classifier = nn.Sequential(
                nn.Linear(input_dim, num_classes),
                nn.Softmax(dim=1)  # 添加 softmax！
            ).to(self.ctx.device)
        else:
            # Multi-layer MLP
            layers = []
            # ... (hidden layers)
            layers.append(nn.Linear(hidden_dim, num_classes))
            layers.append(nn.Softmax(dim=1))  # 添加 softmax！
            self.mlp_classifier = nn.Sequential(*layers).to(self.ctx.device)
```

### 修复 3：训练参数检查

确保配置文件中的参数与源码一致：

**Office-Home**（源码第223行）：
```yaml
train:
  local_update_steps: 1  # 只训练1个epoch！
  optimizer:
    type: 'Adam'
    lr: 0.001
dataloader:
  batch_size: 16
federate:
  total_round_num: 50
```

**Digits**（源码训练循环）：
```yaml
train:
  local_update_steps: 10  # 10个epoch
  optimizer:
    type: 'Adam'
    lr: 0.01
dataloader:
  batch_size: 16
federate:
  total_round_num: 50
```

---

## 修复优先级

### 🔴 立即修复（影响巨大）：
1. **特征值缩放顺序**（问题1） - 这个错误会完全改变数据增强的效果
2. **模型输出层**（问题2） - 影响训练动态和收敛

### 🟡 检查确认：
3. 训练参数（local_epochs, optimizer, lr, batch_size）
4. 数据增强参数（num_per_sample, target_size）

---

## 预期影响

修复这些问题后，准确率应该能达到论文水平：

**Office-Home**（论文 Table 4）：
- Art: ~65%
- Clipart: ~52%
- Product: ~77%
- Real_World: ~79%
- Average: ~68%

**Digits**（论文 Table 4）：
- MNIST: ~97%
- USPS: ~94%
- SVHN: ~63%
- SYN: ~75%
- Average: ~82%

**PACS**（论文 Table 3）：
- Photo: ~95%
- Art_Painting: ~84%
- Cartoon: ~83%
- Sketch: ~82%
- Average: ~86%
