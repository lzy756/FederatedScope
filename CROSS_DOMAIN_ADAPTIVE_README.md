# Cross-Domain Adaptive 联邦学习 - 完整实现

## 项目概述

本项目实现了跨域自适应联邦学习方法，结合了FedLSA（联邦可学习语义锚点）和OnDemFL（按需联邦学习）两种方法，用于处理Office-Caltech-10跨域图像分类任务。

## 核心功能

### 1. 跨域自适应学习 ✅
- **FedLSA**: 超球面对比学习 + 语义锚点对齐
- **OnDemFL**: 分布预测 + 按需客户端选择
- **跨域融合**: 锚点重加权 + 动态客户端采样

### 2. 服务器端均衡测试 ✅
- 四个域（Amazon, Webcam, DSLR, Caltech）独立测试集
- 每个测试集内部类别均衡分布
- 分域准确率 + 加权平均准确率

### 3. 完整的配置系统 ✅
- 所有配置项正确注册
- 灵活的超参数设置
- 完善的验证机制

## 快速开始

### 环境准备

```bash
# 安装依赖
pip install numpy scipy scikit-learn pyyaml protobuf grpcio torch torchvision

# 设置开发模式（已设置）
# DEV_MODE = True in federatedscope/main.py
```

### 运行训练

```bash
# 基本运行
python federatedscope/main.py --cfg scripts/example_configs/cross_domain_adaptive_office_caltech.yaml

# 修改数据集路径（如果需要）
python federatedscope/main.py \
  --cfg scripts/example_configs/cross_domain_adaptive_office_caltech.yaml \
  data.root /your/path/to/office_caltech_10
```

### 运行测试

```bash
# 测试配置加载
python test_config_final.py

# 测试Worker加载
python test_worker_loading.py

# 测试服务器端评估
python test_server_eval.py
```

## 配置说明

### 关键配置项

```yaml
# 联邦学习设置
federate:
  mode: 'standalone'
  client_num: 20
  total_round_num: 300
  sample_client_num: 5
  method: 'cross_domain_adaptive'
  make_global_eval: True  # 启用服务器端评估

# 数据设置
data:
  root: '/root/data/office_caltech_10'
  type: 'office_caltech'
  server_test_samples_per_class: 10  # 服务器测试集每类样本数

# 模型设置
model:
  type: 'fedlsa_cnn'
  num_classes: 10

# Trainer设置
trainer:
  type: 'cross_domain_adaptive'

# FedLSA设置
fedlsa:
  use: True
  lambda_com: 0.5
  tau: 0.1
  use_projector: True
  alpha_sep: 0.6

# OnDemFL设置
ondemfl:
  enable: True
  pretrain_rounds: 150
  ondemand_rounds: 150
  subset_size: 10
  weight_scheme: 'ratio_times_size'

# CrossDomainAdaptive设置
cross_domain_adaptive:
  anchor_reweight: True
  anchor_weight_momentum: 0.6
```

## 输出示例

### 训练过程输出

```
================================================================================
[Round 150] Server-side Evaluation on Balanced Test Sets
================================================================================
--------------------------------------------------------------------------------
Domain-specific Test Results:
--------------------------------------------------------------------------------
  amazon      : Acc = 0.7500 | Samples = 100 | Weight = 0.250
  caltech     : Acc = 0.8200 | Samples = 100 | Weight = 0.250
  dslr        : Acc = 0.7800 | Samples = 100 | Weight = 0.250
  webcam      : Acc = 0.7600 | Samples = 100 | Weight = 0.250
--------------------------------------------------------------------------------
  Weighted Avg: Acc = 0.7775 | Total Samples = 400
================================================================================
```

## 项目结构

```
FederatedScope/
├── federatedscope/
│   ├── main.py                           # 主入口（已修复）
│   ├── core/
│   │   ├── configs/
│   │   │   ├── cfg_model.py             # 模型配置（已添加num_classes）
│   │   │   ├── cfg_data.py              # 数据配置（已添加server_test_samples_per_class）
│   │   │   └── cfg_fedlsa.py            # FedLSA配置（已修复）
│   │   └── workers/
│   │       └── server.py                # 基础Server类
│   ├── contrib/
│   │   ├── configs/
│   │   │   ├── ondemfl.py               # OnDemFL配置
│   │   │   └── cross_domain_adaptive.py # 跨域自适应配置
│   │   ├── worker/
│   │   │   ├── ondemfl.py               # OnDemFL Worker
│   │   │   └── cross_domain_adaptive.py # 跨域自适应Worker（已修复）
│   │   └── trainer/
│   │       └── cross_domain_adaptive_trainer.py # 跨域自适应Trainer
│   └── cv/
│       ├── model/
│       │   └── fedlsa_cnn.py            # FedLSA CNN模型
│       └── dataset/
│           └── office_caltech.py        # Office-Caltech数据集
├── scripts/
│   └── example_configs/
│       └── cross_domain_adaptive_office_caltech.yaml  # 配置文件
├── test_config_final.py                  # 配置测试脚本
├── test_worker_loading.py                # Worker测试脚本
├── test_server_eval.py                   # 评估测试脚本
├── CONFIG_FIX_SUMMARY.md                 # 配置修复总结
├── CONFIG_VERIFICATION_REPORT.md         # 配置验证报告
├── SERVER_DATA_FIX.md                    # 服务器数据修复说明
├── SERVER_EVAL_IMPLEMENTATION.md         # 服务器评估实现说明
└── SERVER_EVAL_USAGE_GUIDE.md           # 服务器评估使用指南
```

## 已解决的问题

### 1. 配置项缺失 ✅
- [x] `model.num_classes` 配置项
- [x] `data.server_test_samples_per_class` 配置项

### 2. 配置初始化 ✅
- [x] `cfg_fedlsa.py` 初始化时访问不存在的配置项
- [x] 配置加载顺序问题

### 3. 服务器数据初始化 ✅
- [x] standalone模式下服务器无数据导致断言失败
- [x] 服务器端测试数据加载

### 4. 依赖问题 ✅
- [x] protobuf版本兼容性
- [x] DEV_MODE设置

## 文档

### 使用文档
- **SERVER_EVAL_USAGE_GUIDE.md**: 服务器端评估使用指南
- **CONFIG_VERIFICATION_REPORT.md**: 配置验证报告

### 实现文档
- **SERVER_EVAL_IMPLEMENTATION.md**: 服务器端评估实现说明
- **SERVER_DATA_FIX.md**: 服务器数据问题修复说明
- **CONFIG_FIX_SUMMARY.md**: 配置修复总结

## 测试脚本

| 脚本 | 功能 | 用途 |
|------|------|------|
| `test_config_final.py` | 配置加载测试 | 验证所有配置项正确注册 |
| `test_worker_loading.py` | Worker加载测试 | 验证Worker类正确注册 |
| `test_server_eval.py` | 评估逻辑测试 | 验证服务器端评估实现 |
| `quick_verify_config.py` | 快速配置验证 | 不加载完整模块的配置验证 |

## 关键特性

### FedLSA（联邦可学习语义锚点）
- 超球面投影器
- 语义锚点学习
- 紧凑性损失 + 分离损失
- 锚点-客户端对齐

### OnDemFL（按需联邦学习）
- 基于梯度的分布预测器
- NNLS客户端选择
- 目标分布匹配
- 两阶段训练（预训练+按需）

### CrossDomainAdaptive（跨域自适应）
- 锚点重加权
- 动态权重更新
- 分域服务器端评估
- 加权平均准确率

## 性能优化

### 评估频率
```yaml
eval:
  freq: 10  # 每10轮评估一次
```

### 测试集大小
```yaml
data:
  server_test_samples_per_class: 10  # 调整此值平衡评估精度和速度
```

### 客户端采样
```yaml
federate:
  sample_client_num: 5  # 每轮采样的客户端数量
```

## 常见问题

### Q1: 配置项不存在
**问题**: `KeyError: 'Non-existent config key: xxx'`
**解决**: 检查对应的配置文件是否已注册该配置项

### Q2: 服务器数据断言失败
**问题**: `AssertionError at server.py:136`
**解决**: 确保使用修复后的 `CrossDomainAdaptiveServer`

### Q3: 数据集路径错误
**问题**: `FileNotFoundError: Domain directory not found`
**解决**: 修改配置文件中的 `data.root` 为正确路径

## 开发团队

- 跨域自适应方法实现
- 服务器端评估逻辑
- 配置系统修复
- 文档编写

## 许可证

Apache License 2.0

## 致谢

基于 FederatedScope 框架开发
- FedLSA: Federated Learning with Learnable Semantic Anchors
- OnDemFL: On-Demand Federated Learning

## 版本历史

- **v1.0.0** (2025-01-13)
  - ✅ 完整的跨域自适应实现
  - ✅ 服务器端均衡测试
  - ✅ 所有配置问题修复
  - ✅ 完整的文档系统
