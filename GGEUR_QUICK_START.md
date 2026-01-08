# GGEUR Implementation - Quick Start Guide

## 完成情况

✅ **已完成核心组件**：

### 基础工具模块
- ✅ `federatedscope/contrib/utils/clip_extractor.py` - CLIP 特征提取器
- ✅ `federatedscope/contrib/utils/geometry_utils.py` - 几何形状计算工具
- ✅ `federatedscope/contrib/utils/covariance_aggregator.py` - 协方差矩阵聚合器

### 核心算法
- ✅ `federatedscope/contrib/data_augmentation/ggeur_augmentation.py` - GGEUR 数据增强（Algorithm 1 & 2）

### Worker 组件
- ✅ `federatedscope/contrib/worker/ggeur_client.py` - GGEUR 客户端
- ✅ `federatedscope/contrib/worker/ggeur_server.py` - GGEUR 服务器

### 训练器
- ✅ `federatedscope/contrib/trainer/ggeur_trainer.py` - GGEUR 训练器（MLP on embeddings）

### 配置文件
- ✅ `federatedscope/core/configs/cfg_ggeur.py` - GGEUR 配置
- ✅ `scripts/example_configs/ggeur_digits_multi_domain.yaml` - Digits 数据集配置
- ✅ `scripts/example_configs/ggeur_pacs_multi_domain.yaml` - PACS 数据集配置
- ✅ `scripts/example_configs/ggeur_officehome_lds.yaml` - Office-Home-LDS ��置

---

## 安装依赖

在运行 GGEUR 之前，需要安装 CLIP：

```bash
pip install git+https://github.com/openai/CLIP.git
```

---

## 集成步骤

### 1. 注册 GGEUR 组件

在 `federatedscope/main.py` 或初始化脚本中添加：

```python
# 导入 GGEUR_Clip 配置
from federatedscope.core.configs.cfg_ggeur import extend_ggeur_cfg

# 在配置初始化后
cfg = extend_ggeur_cfg(cfg)

# 导入并注册 worker 和 trainer
from federatedscope.contrib.worker.ggeur_client import GGEURClient
from federatedscope.contrib.worker.ggeur_server import GGEURServer
from federatedscope.contrib.trainer.ggeur_trainer import GGEURTrainer
from federatedscope.register import register_worker, register_trainer

# 注册
register_worker('ggeur', lambda method: {
    'client': GGEURClient,
    'server': GGEURServer
} if method.lower() == 'ggeur' else None)

register_trainer('ggeur', lambda trainer_type:
    GGEURTrainer if trainer_type.lower() == 'ggeur' else None)
```

### 2. 或者创建独立的注册文件

创建 `federatedscope/contrib/ggeur_register.py`：

```python
"""GGEUR_Clip Registration Module"""

from federatedscope.core.configs.cfg_ggeur import extend_ggeur_cfg
from federatedscope.contrib.worker.ggeur_client import GGEURClient
from federatedscope.contrib.worker.ggeur_server import GGEURServer
from federatedscope.contrib.trainer.ggeur_trainer import GGEURTrainer
from federatedscope.register import register_worker, register_trainer


def register_ggeur():
    """Register all GGEUR_Clip components."""

    # Register worker
    register_worker('ggeur', lambda method: {
        'client': GGEURClient,
        'server': GGEURServer
    } if method.lower() == 'ggeur' else None)

    # Register trainer
    register_trainer('ggeur', lambda trainer_type:
        GGEURTrainer if trainer_type.lower() == 'ggeur' else None)

    print("GGEUR_Clip components registered successfully!")


# Auto-register on import
register_ggeur()
```

然后在 `federatedscope/__init__.py` 中添加：

```python
# Import GGEUR_Clip registration
try:
    from federatedscope.contrib import ggeur_register
except ImportError as e:
    print(f"GGEUR_Clip registration failed: {e}")
```

---

## 运行实验

### Digits 数据集（多域场景）

```bash
python federatedscope/main.py \
    --cfg scripts/example_configs/ggeur_digits_multi_domain.yaml
```

预期结果（参考 Table 4）：
- MNIST: ~97%
- USPS: ~94%
- SVHN: ~63%
- SYN: ~75%
- Average: ~82%

### PACS 数据集

```bash
python federatedscope/main.py \
    --cfg scripts/example_configs/ggeur_pacs_multi_domain.yaml
```

预期结果（参考 Table 6）：
- Photo: ~99.4%
- Art Painting: ~98.4%
- Cartoon: ~99.0%
- Sketch: ~93.6%
- Average: ~97.6%

### Office-Home-LDS 数据集（标签+域偏斜）

```bash
python federatedscope/main.py \
    --cfg scripts/example_configs/ggeur_officehome_lds.yaml
```

预期结果（参考 Table 7, β=0.1）：
- Art: ~79.0%
- Clipart: ~79.0%
- Product: ~90.2%
- Real World: ~88.5%
- Average: ~84.0%

---

## 核心工作流程

### Round 0: 统计量收集与聚合
1. **客户端**：
   - 提取 CLIP 嵌入
   - 计算每个类别的局部协方差矩阵和均值
   - 上传到服务器

2. **服务器**：
   - 收集所有客户端的统计量
   - 聚合计算全局协方差矩阵（Equation 4）
   - 特征值分解得到全局几何形状
   - 下发全局几何形状到客户端
   - （多域）收集并下发域原型

### Round 1+: 增强训练
1. **客户端**：
   - 接收全局几何形状
   - 执行 GGEUR 数据增强：
     - **单域**：Algorithm 1
     - **多域**：Algorithm 2 (Step 1 + Step 2)
   - 在增强后的嵌入上训练 MLP
   - 上传模型参数

2. **服务器**：
   - 聚合客户端模型
   - 正常的联邦平均

---

## 关键参数说明

### GGEUR 配置参数

```yaml
ggeur:
  scenario: 'multi_domain'  # 'single_domain' or 'multi_domain'

  # CLIP 设置
  clip_model: 'ViT-B/16'
  embedding_dim: 512
  cache_embeddings: true  # 缓存嵌入加速

  # 数据增强参数
  n_samples_per_original: 10    # N: 每个原始样本生成数
  m_samples_per_prototype: 500  # M: 每个原型生成数（多域）
  step1_samples_per_class: 500  # 多域 Step 1 目标数

  # MLP 分类器
  mlp_layers: 1           # 层数（1=线性分类器）
  mlp_hidden_dim: 512     # 隐藏层维度
  mlp_dropout: 0.0        # Dropout

  # 工作流
  upload_statistics_round: 0  # 统计量上传轮次
  start_augmentation_round: 1 # 增强开始轮次
```

---

## 调试与测试

### 单元测试

运行各模块的单元测试：

```bash
# 测试几何工具
cd federatedscope/contrib/utils
python geometry_utils.py

# 测试协方差聚合
python covariance_aggregator.py

# 测试 GGEUR_Clip 增强
cd ../data_augmentation
python ggeur_augmentation.py

# 测试 CLIP 提取器
cd ../utils
python clip_extractor.py
```

### 日志调试

设置日志级别为 DEBUG 查看详细信息：

```yaml
# 在配置文件中添加
logging:
  level: DEBUG
```

或在代码中：

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

---

## 故障排查

### 问题 1: CLIP 未安装

**错误**：`ImportError: No module named 'clip'`

**解决**：
```bash
pip install git+https://github.com/openai/CLIP.git
```

### 问题 2: GPU 内存不足

**错误**：`CUDA out of memory`

**解决**：
- 减小 `batch_size`
- 减小 `n_samples_per_original` 或 `m_samples_per_prototype`
- 使用 `cache_embeddings: false` 避免缓存
- 使用 CPU：`use_gpu: False`

### 问题 3: Worker 未注册

**错误**：`Unknown method: ggeur`

**解决**：
确保已导入并注册 GGEUR 组件（见"集成步骤"）

### 问题 4: 数据集未找到

**错误**：`Dataset not found`

**解决**：
- 确保 `data.root` 路径正确
- 下载所需数据集
- 检查数据集名称（如 'Digits', 'PACS', 'OfficeHome'）

---

## 性能优化

### 1. 使用嵌入缓存

```yaml
ggeur:
  cache_embeddings: true  # 第一次提取后缓存
```

### 2. 并行化

- CLIP 提取已自动使用 GPU 并行
- 可调整 `dataloader.num_workers`

### 3. 减少增强数量（快速测试）

```yaml
ggeur:
  n_samples_per_original: 5    # 降低生成数
  m_samples_per_prototype: 200
  step1_samples_per_class: 300
```

---

## 扩展与定制

### 添加新数据集

1. 确保数据集符合 FederatedScope 格式
2. 在配置文件中设置 `data.type`
3. 调整 `model.num_classes`

### 修改 MLP 架构

```yaml
ggeur:
  mlp_layers: 2          # 多层 MLP
  mlp_hidden_dim: 1024   # 更大的隐藏层
  mlp_dropout: 0.1       # 添加 Dropout
```

### 使用不同的 CLIP 模型

```yaml
ggeur:
  clip_model: 'ViT-L/14'  # 更大的模型
  embedding_dim: 768      # 对应的嵌入维度
```

---

## 下一步工作

### 待完成（如需）

1. ⬜ 完整的端到端集成测试
2. ⬜ 与其他 FL 方法（FedProx, SCAFFOLD）的组合
3. ⬜ 单域场景（label skew only）的配置和测试
4. ⬜ 更多数据集的支持和测试
5. ⬜ 性能基准测试和优化
6. ⬜ 详细的 API 文档

### 可选增强

1. ⬜ 差分隐私保护的协方差聚合
2. ⬜ 自适应选择 N 和 M 参数
3. ⬜ 支持其他视觉模型（DINO, MAE）
4. ⬜ 多模态扩展（文本、图像）

---

## 参考资料

- **论文**：D:\Projects\FederatedScope\2503.06457v2_2025-12-03-17_03_33\2503.06457v2.md
- **源代码**：https://github.com/WeiDai-David/2025CVPR_GGEUR
- **工作纲要**：D:\Projects\FederatedScope\GGEUR_IMPLEMENTATION_PLAN.md

---

## 联系与支持

如有问题，请查看：
1. 工作纲要中的详细技术说明
2. 各模块的 docstring 文档
3. 单元测试中的使用示例

---

**创建日期**: 2025-12-11
**状态**: 核心实现完成，可开始测试
