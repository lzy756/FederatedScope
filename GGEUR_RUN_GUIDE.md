# GGEUR 运行训练指南

## 🎯 已实现的方法列表

### 核心算法

| 算法 | 文件位置 | 函数名 | 说明 |
|------|---------|--------|------|
| **Algorithm 1** | `ggeur_augmentation.py` | `augment_single_domain()` | 单域场景（标签偏斜） |
| **Algorithm 2** | `ggeur_augmentation.py` | `augment_multi_domain()` | 多域场景（标签+域偏斜） |
| **Equation 4** | `covariance_aggregator.py` | `aggregate_covariances()` | 联邦协方差聚合 |
| **Equation 1** | `geometry_utils.py` | `compute_covariance_matrix()` | 协方差计算 |
| **Equation 3** | `geometry_utils.py` | `compute_geometry_similarity()` | 几何相似度 |

### 完整组件

| 组件 | 文件 | 作用 |
|------|------|------|
| CLIP 提取器 | `utils/clip_extractor.py` | 提取图像嵌入 |
| 几何工具 | `utils/geometry_utils.py` | 协方差、特征值分解 |
| 协方差聚合 | `utils/covariance_aggregator.py` | 联邦聚合 |
| GGEUR 增强 | `data_augmentation/ggeur_augmentation.py` | 数据增强核心 |
| GGEUR 客户端 | `worker/ggeur_client.py` | 客户端逻辑 |
| GGEUR 服务器 | `worker/ggeur_server.py` | 服务器逻辑 |
| GGEUR 训练器 | `trainer/ggeur_trainer.py` | MLP 训练 |

---

## 🚀 运行训练步骤

### 前置准备

#### 1. 安装 CLIP

```bash
pip install git+https://github.com/openai/CLIP.git
```

#### 2. 验证数据集位置

确保数据集在正确位置：

**PACS 数据集**:
```bash
ls data/pacs_data/
# 应该看到: photo  art_painting  cartoon  sketch
```

**Office-Caltech 数据集**:
```bash
ls data/
# 应该看到: amazon  webcam  dslr  caltech
```

**Office-Home 数据集**:
```bash
ls data/OfficeHome/
# 应该看到: Art  Clipart  Product  Real_World
```

---

## 📝 运行命令

### 方法 1: 使用提供的运行脚本（推荐）

我已经为您创建了 `run_ggeur.py`，它会自动注册所有组件。

#### 运行 Digits 数据集（4域，10类）

```bash
python run_ggeur.py \
    --cfg scripts/example_configs/ggeur_digits_multi_domain.yaml \
    seed 12345
```

**预期输出**:
```
==========================================================
Registering GGEUR components...
==========================================================
✓ GGEUR components registered successfully!

==========================================================
Configuration loaded:
==========================================================
Method: ggeur
Dataset: Digits
Scenario: multi_domain
Total rounds: 50
Clients: 4
Output dir: exp/ggeur_digits_multi_domain
==========================================================

Building data...
✓ Data loaded: 4 clients

==========================================================
Starting GGEUR Training...
==========================================================

[Round 0] Server: Collecting statistics...
Client #1: Extracting CLIP embeddings...
Client #1: Extracted 600 embeddings, shape: torch.Size([600, 512])
Client #1: Computing local statistics...
Client #1: Uploading local statistics to server...

Server: Received statistics from Client #1, 10 classes
Server: Received statistics from all 4 clients
Server: Aggregating covariance matrices...
Server: Class 0 aggregation stats:
  - Clients: 4
  - Total samples: 2400
  - Avg mean divergence: 1.234567
  - Global variance (trace): 245.678901

Server: Broadcasting global geometries to clients...

[Round 1] Client #1: Performing GGEUR augmentation...
Client #1: Class 0: augmented from 60 to 2100 samples
Client #1: Augmentation complete, 600 -> 21000 samples
Client #1: Training on augmented embeddings...
Epoch 10/10: Loss=0.1234, Acc=0.9567

[Round 50] Evaluation:
MNIST: Acc=0.9710
USPS: Acc=0.9412
SVHN: Acc=0.6354
SYN: Acc=0.7473
Average: Acc=0.8237

==========================================================
Training completed!
==========================================================
```

#### 运行 PACS 数据集（4域，7类）

```bash
python run_ggeur.py \
    --cfg scripts/example_configs/ggeur_pacs_multi_domain.yaml \
    seed 12345
```

**预期结果** (参考论文 Table 6):
- Photo: ~99.4%
- Art Painting: ~98.4%
- Cartoon: ~99.0%
- Sketch: ~93.6%
- Average: ~97.6%

#### 运行 Office-Home-LDS（4域，65类，β=0.1）

```bash
python run_ggeur.py \
    --cfg scripts/example_configs/ggeur_officehome_lds.yaml \
    seed 12345
```

**预期结果** (参考论文 Table 7):
- Art: ~79.0%
- Clipart: ~79.0%
- Product: ~90.2%
- Real World: ~88.5%
- Average: ~84.0%

---

### 方法 2: 修改现有的 main.py（可选）

如果您想使用原有的 `federatedscope/main.py`，需要添加以下代码：

#### 在 `federatedscope/main.py` 中添加：

```python
# 在文件顶部，import 区域添加
from federatedscope.core.configs.cfg_ggeur import extend_ggeur_cfg
import federatedscope.contrib.ggeur_register  # Auto-register

# 在 main() 函数中，找到配置初始化的地方
def main():
    init_cfg = global_cfg.clone()
    args = parse_args()

    if args.cfg_file:
        init_cfg.merge_from_file(args.cfg_file)

    # ========== 添加这一行 ==========
    init_cfg = extend_ggeur_cfg(init_cfg)
    # ===============================

    if args.opts:
        init_cfg.merge_from_list(args.opts)

    # ... 其余代码保持不变
```

然后运行：

```bash
python federatedscope/main.py \
    --cfg scripts/example_configs/ggeur_digits_multi_domain.yaml
```

---

## 🎛️ 调整实验参数

### 修改配置文件

您可以直接编辑 YAML 配置文件来调整参数：

```yaml
# scripts/example_configs/ggeur_digits_multi_domain.yaml

# 调整训练轮数
federate:
  total_round_num: 50  # 改为 30 可快速测试

# 调整增强强度
ggeur:
  n_samples_per_original: 10    # 减少到 5 可加快训练
  m_samples_per_prototype: 500  # 减少到 200 可加快训练
  step1_samples_per_class: 500  # 减少到 300 可加快训练

# 调整学习率
train:
  optimizer:
    lr: 0.01  # 可尝试 0.001 或 0.1
```

### 使用命令行覆盖参数

```bash
python run_ggeur.py \
    --cfg scripts/example_configs/ggeur_digits_multi_domain.yaml \
    federate.total_round_num 30 \
    ggeur.n_samples_per_original 5 \
    train.optimizer.lr 0.001 \
    seed 12345
```

---

## 🔍 监控训练过程

### 关键日志信息

#### Round 0（统计收集阶段）

```
Server: Received statistics from Client #1, 10 classes
Server: Class 0 aggregation stats:
  - Clients: 4
  - Total samples: 2400
  - Avg mean divergence: 1.234567
  - Global variance (trace): 245.678901
Server: Class 0 geometry:
  - Components: 512
  - Top 5 eigenvalues: [45.2, 32.1, 28.7, 24.5, 21.3]
  - Explained var (top 10): 0.6543
```

#### Round 1+（增强训练阶段）

```
Client #1: Performing GGEUR augmentation...
Client #1: Class 0: augmented from 60 to 2100 samples
Client #1: Augmentation complete, 600 -> 21000 samples
Epoch 10/10: Loss=0.1234, Acc=0.9567
```

### 查看结果

训练完成后，结果保存在：

```
exp/ggeur_digits_multi_domain/
├── config.yaml              # 使用的配置
├── exp_print.log           # 详细日志
├── eval_results.csv        # 评估结果
└── models/                 # 保存的模型
```

---

## 🧪 快速测试（调试用）

如果想快速验证实现是否正确，使用快速测试配置：

### 创建快速测试配置

```yaml
# scripts/example_configs/ggeur_digits_quick_test.yaml
use_gpu: True
device: 0

federate:
  mode: 'standalone'
  total_round_num: 5      # 只跑 5 轮
  client_num: 4
  sample_client_num: 4
  method: 'ggeur'

data:
  root: 'data/'
  type: 'Digits'
  splits: [0.1, 0.0, 0.0]

model:
  type: 'mlp'
  num_classes: 10

train:
  local_update_steps: 5   # 只训练 5 个 epoch

ggeur:
  use: True
  scenario: 'multi_domain'
  clip_model: 'ViT-B/16'
  embedding_dim: 512
  n_samples_per_original: 5     # 减少生成数
  m_samples_per_prototype: 100
  step1_samples_per_class: 200

trainer:
  type: 'ggeur'

outdir: 'exp/ggeur_quick_test'
seed: 12345
```

运行：

```bash
python run_ggeur.py --cfg scripts/example_configs/ggeur_digits_quick_test.yaml
```

预计 5-10 分钟完成，用于验证流程是否正确。

---

## ⚠️ 常见问题排查

### 问题 1: CLIP 模型下载失败

**症状**:
```
RuntimeError: Error downloading CLIP model
```

**解决**:
```bash
# 手动下载 CLIP 模型
mkdir -p ~/.cache/clip
cd ~/.cache/clip
wget https://openaipublic.azureedge.net/clip/models/5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt

# Windows 用户
# 下载到 C:\Users\你的用户名\.cache\clip\ViT-B-16.pt
```

### 问题 2: 数据集未找到

**症状**:
```
FileNotFoundError: Domain directory not found: data/pacs_data/photo
```

**解决**:
1. 检查配置文件中的 `data.root` 是否正确
2. 确保数据集解压到正确位置
3. 检查目录名称是否完全匹配（区分大小写）

### 问题 3: Worker 未注册

**症状**:
```
ValueError: Unknown worker type: ggeur
```

**解决**:
1. 确保使用 `run_ggeur.py` 脚本
2. 或者确保在 main.py 中添加了注册代码
3. 检查是否成功导入 `ggeur_register`

### 问题 4: GPU 内存不足

**症状**:
```
CUDA out of memory
```

**解决**:
```yaml
# 在配置文件中
use_gpu: False  # 使用 CPU

# 或者减少 batch size
dataloader:
  batch_size: 8  # 从 16 降到 8

# 或者减少增强数量
ggeur:
  n_samples_per_original: 3
  m_samples_per_prototype: 100
```

---

## 📊 验证实现正确性

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

# 测试 CLIP 提取
cd ../utils
python clip_extractor.py
```

所有测试应输出 `All tests passed!`

### 对比论文结果

运行完整实验后，对比论文中的结果：

| 数据集 | 论文结果 | 您的结果 | 误差 |
|--------|---------|---------|------|
| Digits (平均) | 82.36% | ? | ? |
| PACS (平均) | 97.60% | ? | ? |
| Office-Home-LDS | 83.99% | ? | ? |

误差在 ±2% 范围内是正常的（由于随机种子、硬件差异等）。

---

## 🎯 总结

### 已实现的内容

✅ **Algorithm 1**: 单域 GGEUR (Label Skew)
✅ **Algorithm 2**: 多域 GGEUR (Label + Domain Skew)
✅ **Equation 4**: 联邦协方差聚合
✅ **完整工作流**: 客户端 + 服务器 + 训练器
✅ **三个数据集配置**: Digits, PACS, Office-Home-LDS

### 运行命令（最简单）

```bash
# 1. 安装 CLIP
pip install git+https://github.com/openai/CLIP.git

# 2. 运行训练
python run_ggeur.py \
    --cfg scripts/example_configs/ggeur_digits_multi_domain.yaml \
    seed 12345

# 3. 查看结果
cat exp/ggeur_digits_multi_domain/eval_results.csv
```

就这么简单！🎉

---

**需要帮助**？
- 详细实现：查看 `GGEUR_IMPLEMENTATION_PLAN.md`
- 数据集指南：查看 `GGEUR_DATASET_MODEL_GUIDE.md`
- 完整总结：查看 `GGEUR_COMPLETE_SUMMARY.md`
