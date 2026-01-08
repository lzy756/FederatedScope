# GGEUR 数据集和模型完整说明

## 📦 模型说明

### 1. CLIP 模型（预训练特征提取器）

**来源**: OpenAI 官方预训练模型

**自动下载位置**:
- Linux/Mac: `~/.cache/clip/ViT-B-16.pt` (~350MB)
- Windows: `C:\Users\你的用户名\.cache\clip\ViT-B-16.pt`

**下载方式**:
```python
# 首次运行时自动下载，无需手动操作
import clip
model, preprocess = clip.load('ViT-B/16', device='cuda')
```

**作用**:
- 将图像转换为 512 维嵌入向量
- **不需要训练**，直接使用预训练权重
- 在联邦学习中所有客户端共享相同的 CLIP 模型

### 2. MLP 分类器（联邦学习模型）

**定义位置**: `federatedscope/contrib/trainer/ggeur_trainer.py`

**架构**:
```python
# 单层 MLP (默认)
Linear(512 -> num_classes)

# 多层 MLP (可选)
Linear(512 -> hidden_dim) -> ReLU -> Linear(hidden_dim -> num_classes)
```

**特点**:
- 非常轻量（只有几千个参数）
- 在 CLIP 嵌入空间训练，不在原始图像空间
- 这是联邦学习要训练和聚合的模型

---

## 📊 数据集说明

### FederatedScope 数据集结构

您的项目中已有数据集加载器：
- `federatedscope/cv/dataset/pacs.py` - PACS 数据集
- `federatedscope/cv/dataset/office_caltech.py` - Office-Caltech 数据集

### 预期数据集目录结构

#### 1. PACS 数据集

**下载地址**: https://drive.google.com/drive/folders/0B6x7gtvErXgfUU1WcGRpdUN3Y1U

**目录结构**:
```
data/
└── pacs_data/
    ├── photo/
    │   ├── dog/
    │   │   ├── pic_001.jpg
    │   │   └── ...
    │   ├── elephant/
    │   └── ... (7 classes)
    ├── art_painting/
    │   └── ... (same 7 classes)
    ├── cartoon/
    │   └── ... (same 7 classes)
    └── sketch/
        └── ... (same 7 classes)
```

**7 个类别**: dog, elephant, giraffe, guitar, horse, house, person

**4 个域**: photo, art_painting, cartoon, sketch

#### 2. Office-Caltech 数据集

**下载地址**:
- Office: https://faculty.cc.gatech.edu/~judy/domainadapt/
- Caltech: http://www.vision.caltech.edu/Image_Datasets/Caltech256/

**目录结构**:
```
data/
├── amazon/
│   ├── back_pack/
│   ├── bike/
│   └── ... (10 classes)
├── webcam/
│   └── ... (10 classes)
├── dslr/
│   └── ... (10 classes)
└── caltech/
    └── ... (10 classes)
```

**10 个类别**: back_pack, bike, calculator, headphones, keyboard, laptop_computer, monitor, mouse, mug, projector

**4 个域**: amazon, webcam, dslr, caltech

#### 3. Digits 数据集

**说明**: 需要检查 FederatedScope 是否已有加载器，或者需要实现

**可能的数据集**:
- MNIST: 自动下载（torchvision）
- USPS: 需要下载
- SVHN: 自动下载（torchvision）
- SYN: 合成数据，需要生成脚本

#### 4. Office-Home 数据集

**下载地址**: http://hemanthdv.org/OfficeHome-Dataset/

**目录结构**:
```
data/
└── OfficeHome/
    ├── Art/
    │   ├── Alarm_Clock/
    │   └── ... (65 classes)
    ├── Clipart/
    ├── Product/
    └── Real_World/
```

**65 个类别**, **4 个域**

---

## 🔧 我的实现如何工作

### 数据流

```
原始图像 → CLIP提取 → 嵌入(512维) → GGEUR增强 → MLP训练
  ↓           ↓              ↓              ↓           ↓
 PACS    clip_extractor   几何形状    augmentation  trainer
```

### 关键点

1. **不修改数据集加载器**: 复用 FederatedScope 现有的数据集类
2. **CLIP 是工具**: 用于提取特征，不参与联邦学习
3. **MLP 是模型**: 这才是要训练和聚合的联邦学习模型

---

## 🚀 集成到 FederatedScope

### 需要的修改

#### 1. 修改 `federatedscope/main.py` 或创建 `run_ggeur.py`

```python
#!/usr/bin/env python
"""
Run GGEUR_Clip experiments
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from federatedscope.core.cmd_args import parse_args
from federatedscope.core.auxiliaries.data_builder import get_data
from federatedscope.core.auxiliaries.worker_builder import get_client_cls, get_server_cls
from federatedscope.core.auxiliaries.runner_builder import get_runner
from federatedscope.core.configs.config import global_cfg

# Import GGEUR_Clip components
from federatedscope.contrib.ggeur_register import extend_ggeur_cfg
import federatedscope.contrib.ggeur_register  # Auto-register


def main():
    # Parse arguments
    init_cfg = global_cfg.clone()
    args = parse_args()

    # Merge config file
    if args.cfg_file:
        init_cfg.merge_from_file(args.cfg_file)

    # Extend with GGEUR_Clip config
    init_cfg = extend_ggeur_cfg(init_cfg)

    # Merge command line args
    if args.opts:
        init_cfg.merge_from_list(args.opts)

    init_cfg.freeze()

    # Build data, workers, and runner
    data, modified_cfg = get_data(init_cfg.clone())

    # Run
    runner = get_runner(data=data, config=modified_cfg.clone())
    runner.run()


if __name__ == '__main__':
    main()
```

#### 2. 或者修改现有的 `federatedscope/main.py`

在文件顶部添加：

```python
# 在 imports 之后，main() 之前
from federatedscope.contrib.ggeur_register import extend_ggeur_cfg
import federatedscope.contrib.ggeur_register  # Auto-register

# 在配置初始化之后
def main():
    init_cfg = global_cfg.clone()
    # ... 其他代码 ...

    # 添加这一行
    init_cfg = extend_ggeur_cfg(init_cfg)

    # ... 继续其他代码 ...
```

#### 3. 确保数据集在正确位置

根据配置文件中的 `data.root` 设置，确保数据集在正确位置：

```yaml
# 在配置文件中
data:
  root: 'data/'  # 数据集根目录
  type: 'PACS'   # 或 'OfficeCaltech', 'Digits', 'OfficeHome'
```

对应的数据集应该在：
- PACS: `data/pacs_data/`
- Office-Caltech: `data/amazon/`, `data/webcam/` 等
- Office-Home: `data/OfficeHome/`

---

## 📥 数据集下载指南

### PACS 数据集

```bash
# 1. 从 Google Drive 下载
# https://drive.google.com/drive/folders/0B6x7gtvErXgfUU1WcGRpdUN3Y1U

# 2. 解压到正确位置
unzip pacs.zip -d data/
# 确保结构是 data/pacs_data/photo/...
```

### Office-Caltech 数据集

```bash
# 可能需要手动下载和整理
# 或者查看是否有下载脚本：
python scripts/download_office_caltech.py  # 如果存在
```

### Digits 数据集

MNIST 和 SVHN 可以通过 torchvision 自动下载，但 USPS 和 SYN 可能需要额外处理。

---

## 🎯 测试流程

### 1. 简单测试（验证组件）

```bash
# 测试各个模块
cd federatedscope/contrib/utils
python geometry_utils.py
python covariance_aggregator.py
python clip_extractor.py

cd ../data_augmentation
python ggeur_augmentation.py
```

### 2. 端到端测试

```bash
# 创建运行脚本
python run_ggeur.py \
    --cfg scripts/example_configs/ggeur_pacs_multi_domain.yaml \
    seed 12345
```

---

## ⚠️ 常见问题

### 问题 1: 数据集未找到

**错误**: `FileNotFoundError: Domain directory not found`

**解决**:
1. 检查 `data.root` 配置
2. 确保数据集下载并解压到正确位置
3. 检查目录结构是否匹配

### 问题 2: CLIP 下载失败

**错误**: `Connection timeout`

**解决**:
```bash
# 手动下载 CLIP 模型
# 从 https://openaipublic.azureedge.net/clip/models/
# 下载 ViT-B-16.pt
# 放到 ~/.cache/clip/
```

### 问题 3: Worker 未注册

**错误**: `Unknown worker type: ggeur`

**解决**: 确保在 main.py 中导入了 `ggeur_register`

---

## 📝 总结

**我没有修改任何现有代码**，所有 GGEUR 功能都是**插件式添加**：

✅ **新增文件**: 14 个文件，全部在 `contrib/` 目录
✅ **使用现有数据集**: 复用 FederatedScope 的数据加载器
✅ **CLIP 自动下载**: 首次运行时自动获取
✅ **MLP 动态构建**: 在训练器中根据配置构建

**您只需要**:
1. 下载数据集到 `data/` 目录
2. 安装 CLIP: `pip install git+https://github.com/openai/CLIP.git`
3. 在 main.py 中添加 2 行代码导入 GGEUR
4. 运行实验

就这么简单！🎉
