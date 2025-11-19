# 配置项缺失问题修复总结

## 问题描述

运行训练时遇到错误：
```
KeyError: 'Non-existent config key: data.server_test_samples_per_class'
```

## 原因

配置文件中使用了 `data.server_test_samples_per_class` 配置项，但该配置项没有在配置系统中注册。

## 解决方案

在 `federatedscope/core/configs/cfg_data.py` 中添加配置项注册。

### 修改位置

**文件**: `federatedscope/core/configs/cfg_data.py`
**行号**: 33-35

### 修改内容

```python
# Server-side test data configuration
cfg.data.server_test_samples_per_class = 10  # Number of samples per class
# for server-side balanced test sets (used in cross-domain evaluation)
```

### 完整的配置项注册

```python
def extend_data_cfg(cfg):
    # ... 其他配置 ...

    cfg.data.dirichlet_alpha = 0.0  # Dirichlet concentration parameter

    # Server-side test data configuration
    cfg.data.server_test_samples_per_class = 10  # Number of samples per class
    # for server-side balanced test sets (used in cross-domain evaluation)

    cfg.data.transform = [...]
    # ... 其他配置 ...
```

## 验证

运行测试脚本：
```bash
python test_config_final.py
```

预期输出：
```
================================================================================
测试配置文件加载
================================================================================

[1] 全局配置导入成功
[2] 配置克隆成功

[3] 检查配置项:
  - model.num_classes: 10
  - data.server_test_samples_per_class: 10

[4] 加载配置文件...
  [OK] 配置文件加载成功

[5] 验证关键配置:
  - federate.method: cross_domain_adaptive
  - federate.make_global_eval: True
  - trainer.type: cross_domain_adaptive
  - model.type: fedlsa_cnn
  - model.num_classes: 10
  - data.type: office_caltech
  - data.server_test_samples_per_class: 10
  - fedlsa.use: True
  - ondemfl.enable: True
  - cross_domain_adaptive.anchor_reweight: True

[OK] 所有配置项验证通过!

================================================================================
配置测试通过! 可以开始训练了。
================================================================================
```

## 所有已修复的配置问题

本项目中已修复的所有配置相关问题：

### 1. model.num_classes 配置项缺失 ✅

**文件**: `federatedscope/core/configs/cfg_model.py:20`

```python
cfg.model.num_classes = 10  # Number of classes for classification tasks
```

### 2. data.server_test_samples_per_class 配置项缺失 ✅

**文件**: `federatedscope/core/configs/cfg_data.py:34`

```python
cfg.data.server_test_samples_per_class = 10  # Server-side balanced test set
```

### 3. cfg_fedlsa.py 初始化问题 ✅

**文件**: `federatedscope/core/configs/cfg_fedlsa.py:72-77`

移除了在配置初始化时访问 `cfg.model.num_classes` 的代码，改为注释说明。

### 4. CrossDomainAdaptiveServer 数据初始化问题 ✅

**文件**: `federatedscope/contrib/worker/cross_domain_adaptive.py:28-42`

添加了在 `make_global_eval=True` 但 `data=None` 时创建占位数据的逻辑。

## 配置文件状态

**配置文件**: `scripts/example_configs/cross_domain_adaptive_office_caltech.yaml`

### 关键配置项

```yaml
federate:
  method: 'cross_domain_adaptive'
  make_global_eval: True

data:
  type: 'office_caltech'
  server_test_samples_per_class: 10  # ← 新增

model:
  type: 'fedlsa_cnn'
  num_classes: 10  # ← 已注册

trainer:
  type: 'cross_domain_adaptive'

fedlsa:
  use: True

ondemfl:
  enable: True

cross_domain_adaptive:
  anchor_reweight: True
```

## 测试清单

- [x] model.num_classes 配置项注册
- [x] data.server_test_samples_per_class 配置项注册
- [x] cfg_fedlsa.py 初始化修复
- [x] CrossDomainAdaptiveServer 数据初始化修复
- [x] 配置文件加载测试
- [x] Worker 类加载测试
- [x] 服务器端评估逻辑实现

## 现在可以运行的命令

```bash
# 测试配置加载
python test_config_final.py

# 测试Worker加载
python test_worker_loading.py

# 测试服务器端评估
python test_server_eval.py

# 运行训练（如果数据集已准备）
python federatedscope/main.py --cfg scripts/example_configs/cross_domain_adaptive_office_caltech.yaml
```

## 总结

所有配置项缺失问题已经修复：
- ✅ 模型配置项已注册
- ✅ 数据配置项已注册
- ✅ 配置初始化逻辑已修复
- ✅ 服务器数据初始化已修复
- ✅ 配置文件可以正常加载
- ✅ 所有组件可以正常实例化

系统现在已经完全就绪，可以开始训练了！
