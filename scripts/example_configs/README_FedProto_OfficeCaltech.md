# FedProto Office-Caltech-10 Configuration Guide

本目录包含多个FedProto方法在Office-Caltech-10数据集上的配置文件，用于域偏斜的联邦学习场景。

## 数据集说明

**Office-Caltech-10** 是一个经典的域适应数据集，包含4个域：
- **Amazon** (A): 电商图片
- **Webcam** (W): 网络摄像头图片
- **DSLR** (D): 单反相机图片
- **Caltech** (C): Caltech-256数据集子集

每个域包含10个类别的物体图片，具有显著的域偏移（domain shift）。

## 配置文件说明

### 1. `fedproto_office_caltech_v2.yaml` ⭐ **推荐基线**

**特点：**
- 使用余弦距离 + 原型归一化
- 包含投影层（512 → 256 → 512）
- 加权平均聚合（基于样本数）
- 完整的训练配置（学习率调度、梯度裁剪等）

**适用场景：** 标准实验基线，与FedLSA对比

**运行命令：**
```bash
python federatedscope/main.py --cfg scripts/example_configs/fedproto_office_caltech_v2.yaml
```

**关键参数：**
```yaml
fedproto:
  proto_weight: 1.0           # 原型损失权重
  distance_metric: 'cosine'   # 余弦距离
  temperature: 0.1            # 温度参数
  normalize_prototypes: True  # 归一化到单位球面
  use_projector: True         # 使用投影层
```

---

### 2. `fedproto_office_caltech_euclidean.yaml` 🚀 **快速基线**

**特点：**
- 使用欧氏距离（无归一化）
- 不使用投影层（更简单）
- 更快的训练速度

**适用场景：** 快速实验、消融研究

**运行命令：**
```bash
python federatedscope/main.py --cfg scripts/example_configs/fedproto_office_caltech_euclidean.yaml
```

**关键参数：**
```yaml
fedproto:
  proto_weight: 1.0
  distance_metric: 'euclidean'
  use_projector: False
  normalize_prototypes: False
```

---

### 3. `fedproto_office_caltech_strong.yaml` 💪 **强正则化**

**特点：**
- 更大的原型损失权重（2.0）
- 更低的温度参数（0.05）
- 适合高度异构数据

**适用场景：** 数据高度非IID、域偏移严重时

**运行命令：**
```bash
python federatedscope/main.py --cfg scripts/example_configs/fedproto_office_caltech_strong.yaml
```

**关键参数：**
```yaml
fedproto:
  proto_weight: 2.0           # 更强的原型约束
  temperature: 0.05           # 更尖锐的分布
```

---

## 配置参数详解

### 联邦学习设置

```yaml
federate:
  client_num: 50              # 总客户端数（分布在4个域）
  sample_client_num: 10       # 每轮采样客户端数
  total_round_num: 100        # 总通信轮数
  share_local_model: True     # 共享模型参数
```

### 数据设置

```yaml
data:
  root: '/home/liziyu/data/office_caltech_10'
  type: 'office_caltech'
  splits: [0.8, 0.1, 0.1]     # 训练/验证/测试比例
  dirichlet_alpha: 0.5        # 控制非IID程度（越小越异构）
```

### 模型设置

```yaml
model:
  type: 'fedlsa_cnn'          # 使用FedLSA的CNN架构
  hidden: 512                 # 隐藏层维度
  num_classes: 10             # 类别数
```

### FedProto核心参数

| 参数 | 说明 | 推荐值 | 取值范围 |
|------|------|--------|----------|
| `proto_weight` | 原型损失权重λ | 1.0 | 0.5-2.0 |
| `embedding_dim` | Embedding维度 | 512 | 与model.hidden一致 |
| `distance_metric` | 距离度量 | cosine | euclidean/cosine |
| `temperature` | 温度参数（cosine） | 0.1 | 0.05-0.5 |
| `aggregation_method` | 聚合方法 | weighted_mean | mean/weighted_mean |
| `normalize_prototypes` | 原型归一化 | True | True/False |
| `use_projector` | 使用投影层 | True | True/False |

---

## 实验对比建议

### 1. 与FedAvg对比

修改配置：
```yaml
trainer:
  type: 'general'  # 改为通用trainer

# 注释掉fedproto部分
# fedproto:
#   use: False
```

### 2. 与FedLSA对比

使用 `fedlsa_office_caltech.yaml`：
```bash
python federatedscope/main.py --cfg scripts/example_configs/fedlsa_office_caltech.yaml
```

### 3. 消融实验

**测试投影层的影响：**
```yaml
fedproto:
  use_projector: False  # vs True
```

**测试距离度量的影响：**
```yaml
fedproto:
  distance_metric: 'euclidean'  # vs 'cosine'
```

**测试聚合方法的影响：**
```yaml
fedproto:
  aggregation_method: 'mean'  # vs 'weighted_mean'
```

**测试原型权重的影响：**
```yaml
fedproto:
  proto_weight: [0.5, 1.0, 2.0, 5.0]
```

---

## 预期结果

根据FedProto论文，在域偏斜场景下的性能提升：

| 方法 | 预期准确率 | 特点 |
|------|-----------|------|
| FedAvg | ~65-70% | 基线方法 |
| FedProto | ~72-78% | 原型学习 |
| FedLSA | ~75-80% | 语义锚点学习 |

**注意：** 实际结果依赖于：
- 数据分布（dirichlet_alpha）
- 采样策略（sample_client_num）
- 超参数调优

---

## 快速开始

### 1. 准备数据

确保Office-Caltech-10数据集在正确位置：
```bash
ls /home/liziyu/data/office_caltech_10
# 应该看到: amazon/ webcam/ dslr/ caltech/
```

### 2. 运行基线实验

```bash
# FedProto (推荐配置)
python federatedscope/main.py --cfg scripts/example_configs/fedproto_office_caltech_v2.yaml

# FedProto (欧氏距离)
python federatedscope/main.py --cfg scripts/example_configs/fedproto_office_caltech_euclidean.yaml

# FedProto (强正则化)
python federatedscope/main.py --cfg scripts/example_configs/fedproto_office_caltech_strong.yaml
```

### 3. 修改超参数（命令行）

```bash
# 修改原型权重
python federatedscope/main.py --cfg scripts/example_configs/fedproto_office_caltech_v2.yaml \
  fedproto.proto_weight 2.0

# 修改客户端采样数
python federatedscope/main.py --cfg scripts/example_configs/fedproto_office_caltech_v2.yaml \
  federate.sample_client_num 5

# 修改学习率
python federatedscope/main.py --cfg scripts/example_configs/fedproto_office_caltech_v2.yaml \
  train.optimizer.lr 0.001

# 组合修改
python federatedscope/main.py --cfg scripts/example_configs/fedproto_office_caltech_v2.yaml \
  fedproto.proto_weight 1.5 \
  fedproto.temperature 0.2 \
  train.optimizer.lr 0.005
```

---

## 调试技巧

### 1. 快速测试（少轮次）

```bash
python federatedscope/main.py --cfg scripts/example_configs/fedproto_office_caltech_v2.yaml \
  federate.total_round_num 5 \
  data.subsample 0.1
```

### 2. 单客户端测试

```bash
python federatedscope/main.py --cfg scripts/example_configs/fedproto_office_caltech_v2.yaml \
  federate.client_num 1 \
  federate.sample_client_num 1
```

### 3. 查看详细日志

```bash
python federatedscope/main.py --cfg scripts/example_configs/fedproto_office_caltech_v2.yaml \
  verbose 2
```

---

## 常见问题

### Q1: 数据路径错误

**错误：** `FileNotFoundError: [Errno 2] No such file or directory: '/home/liziyu/data/office_caltech_10'`

**解决：** 修改配置文件中的 `data.root` 为你的实际路径

### Q2: 显存不足

**解决：**
```yaml
dataloader:
  batch_size: 4        # 减小batch size
train:
  local_update_steps: 3  # 减少本地训练步数
```

### Q3: 原型为None

**原因：** 可能是第一轮训练，或数据集为空

**解决：** 检查日志，确认数据加载成功

### Q4: 性能不佳

**尝试：**
1. 增大 `proto_weight` (1.0 → 2.0)
2. 使用余弦距离 + 归一化
3. 启用投影层
4. 调整温度参数 (0.1 → 0.05)

---

## 输出结果

实验结果保存在：
```
exp_fedproto/office_caltech/fedproto_baseline/
├── config.yaml          # 完整配置
├── exp_print.log        # 运行日志
├── eval_results.csv     # 评估结果
└── models/              # 保存的模型
```

---

## 引用

如果使用FedProto方法，请引用：

```bibtex
@inproceedings{fedproto,
  title={FedProto: Federated Prototype Learning across Heterogeneous Clients},
  author={...},
  booktitle={AAAI},
  year={2022}
}
```

---

## 联系与支持

如有问题，请查看：
- FedProto实现文档: `federatedscope/core/trainers/README_FedProto.md`
- FederatedScope官方文档: https://federatedscope.io
