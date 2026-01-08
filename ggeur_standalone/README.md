# GGEUR Standalone - Office-Home-LDS

这是GGEUR论文的独立实现版本，用于复现Office-Home-LDS实验。

## 目录结构

```
ggeur_standalone/
├── run_ggeur.py                    # 主运行脚本
├── step1_data_preparation.py       # 数据准备（划分训练/测试集，Dirichlet分配）
├── step2_client_class_index.py     # 客户端-类索引匹配
├── step3_clip_feature_extraction.py # CLIP特征提取
├── step4_covariance_aggregation.py  # 协方差矩阵聚合
├── step5_prototype_computation.py   # 原型计算
├── step6_feature_generation.py      # GGEUR特征生成（核心）
├── step7_fedavg_training.py         # FedAvg训练
└── workspace/                       # 运行时输出目录
    ├── output_indices/              # 数据索引
    ├── output_client_class_indices/ # 客户端-类索引
    ├── clip_train_features/         # 训练集CLIP特征
    ├── clip_test_features/          # 测试集CLIP特征
    ├── cov_matrices/                # 协方差矩阵
    ├── prototypes/                  # 类原型
    ├── augmented_features/          # 增强后的特征
    ├── model/                       # 保存的模型
    └── results/                     # 训练日志和图表
```

## 前置条件

### 1. 数据集准备

Office-Home数据集已放置在服务器上：
```
/root/CDA_new/OfficeHomeDataset_10072016/
├── Art/
│   ├── Alarm_Clock/
│   ├── Backpack/
│   └── ... (65个类别)
├── Clipart/
├── Product/
└── Real World/   (注意：可能是"Real World"或"Real_World")
```

### 2. CLIP模型

使用ViT-B-16模型，已放置在服务器上：
```
/root/model/open_clip_vitb16.bin
```

### 3. 依赖安装

```bash
pip install torch torchvision numpy open_clip_torch matplotlib tqdm pillow
```

## 使用方法

### 完整运行（从头开始）

```bash
cd ggeur_standalone
python run_ggeur.py
```

或者指定数据路径：
```bash
python run_ggeur.py --data_path /root/CDA_new/OfficeHomeDataset_10072016
```

### 分步运行

如果只需要运行某些步骤：

```bash
# 只运行数据预处理（步骤1-3）
python run_ggeur.py --steps 1 2 3

# 只运行特征计算（步骤4-5）
python run_ggeur.py --steps 4 5

# 只运行GGEUR特征生成（步骤6）
python run_ggeur.py --steps 6

# 只运行训练（步骤7）
python run_ggeur.py --steps 7
```

### 单独运行每个步骤

```bash
# Step 1: 数据准备
python step1_data_preparation.py

# Step 2: 客户端-类索引匹配
python step2_client_class_index.py

# Step 3: CLIP特征提取
python step3_clip_feature_extraction.py

# Step 4: 协方差矩阵聚合
python step4_covariance_aggregation.py

# Step 5: 原型计算
python step5_prototype_computation.py

# Step 6: GGEUR特征生成
python step6_feature_generation.py

# Step 7: FedAvg训练
python step7_fedavg_training.py
```

## 参数说明

### 数据相关
- `--data_path`: Office-Home数据集路径（默认: /root/CDA_new/OfficeHomeDataset_10072016）
- `--workspace`: 工作目录（默认: ./ggeur_standalone/workspace）
- `--clip_model_path`: CLIP模型路径（默认: /root/model/open_clip_vitb16.bin）
- `--backbone`: CLIP骨干网络（默认: ViT-B-16）
- `--alpha`: Dirichlet分布参数（默认: 0.5）
- `--seed`: 随机种子（默认: 42）

### 特征生成相关
- `--target_size`: 每个类别增强后的样本数（默认: 50）

### 训练相关
- `--communication_rounds`: 联邦通信轮数（默认: 50）
- `--local_epochs`: 每轮本地训练轮数（默认: 1）
- `--batch_size`: 批次大小（默认: 16）
- `--learning_rate`: 学习率（默认: 0.001）

## GGEUR算法流程

1. **数据划分**: 将Office-Home的4个域（Art, Clipart, Product, Real_World）各分配给1个客户端，使用Dirichlet(0.5)分配样本

2. **特征提取**: 使用CLIP ViT-B-32提取512维特征

3. **协方差聚合**: 服务器聚合所有客户端的同类样本，计算全局协方差矩阵

4. **原型计算**: 每个客户端计算每个类的原型（特征均值）

5. **特征生成（GGEUR核心）**:
   - 对于每个客户端的每个类：
     - 如果有原始样本：使用协方差矩阵扩充样本
     - 使用其他客户端的原型生成额外样本
     - 合并原始和生成的样本，达到目标数量

6. **联邦训练**: 使用FedAvg训练线性分类头（512->65）

## 预期结果

论文报告的Office-Home-LDS结果（FedAvg + GGEUR）：
- Art: ~70%
- Clipart: ~65%
- Product: ~80%
- Real_World: ~80%
- 平均: ~74%

## 常见问题

### Q: CLIP模型下载失败怎么办？
A: 可以手动下载模型并指定路径：
```bash
python run_ggeur.py --clip_model_path /path/to/open_clip_vitb32.bin
```

### Q: 显存不足怎么办？
A: 减小batch_size：
```bash
python run_ggeur.py --batch_size 8
```

### Q: 想要更高的精度怎么办？
A: 增加通信轮数和target_size：
```bash
python run_ggeur.py --communication_rounds 100 --target_size 100
```

## 与原论文代码的对应关系

| 本实现 | 原代码 |
|--------|--------|
| step1_data_preparation.py | Dataset-Office-Home-LDS.py |
| step2_client_class_index.py | client_class_cross_index.py |
| step3_clip_feature_extraction.py | train_client_class_clip_features2tensor.py + test_clip_features2tensor.py |
| step4_covariance_aggregation.py | clip_tensor2aggregate_covariance_matrix.py |
| step5_prototype_computation.py | prototype_clip_features2tensor.py |
| step6_feature_generation.py | prototype_cov_matrix_generate_features.py |
| step7_fedavg_training.py | FedAvg_GGEUR.py |
