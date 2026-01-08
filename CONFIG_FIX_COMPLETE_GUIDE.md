# FederatedScope 配置错误修复指南

## 问题诊断

你遇到的错误是：
```
KeyError: 'Non-existent config key: data.server_test_samples_per_class'
```

**根本原因：**
从错误信息可以看出，你的服务器上有两个不同的 FederatedScope 代码库：
- `/root/CDA/federatedscope/` - 你运行脚本的位置
- `/root/FedLSA/federatedscope/` - Python 实际导入模块的位置 ⚠️

即使你在 `/root/CDA` 修改了配置文件，Python 仍然从 `/root/FedLSA` 导入模块，导致修改无效。

## 解决方案

### 方案 1：自动诊断和修复（推荐）

1. **上传脚本到服务器**
   将以下文件上传到服务器：
   - `diagnose_import_path.py` - 诊断工具
   - `auto_fix_config.sh` - 自动修复脚本

2. **运行诊断**
   ```bash
   cd /root/CDA  # 或你的工作目录
   python diagnose_import_path.py
   ```

   诊断脚本会告诉你：
   - Python 实际使用的代码库路径
   - 哪些配置文件缺失或不正确
   - 所有 FederatedScope 安装位置

3. **自动修复**
   ```bash
   # 自动检测并修复
   bash auto_fix_config.sh

   # 或手动指定要修复的目录
   bash auto_fix_config.sh /root/FedLSA
   ```

### 方案 2：手动修复正确的目录

1. **确定 Python 使用的目录**
   ```bash
   python -c "import federatedscope; print(federatedscope.__file__)"
   ```

   这会显示类似：`/root/FedLSA/federatedscope/__init__.py`

   那么你需要修改的是 `/root/FedLSA/` 目录！

2. **在正确的目录修改配置文件**

   **文件 1：`/root/FedLSA/federatedscope/core/configs/cfg_data.py`**

   找到第 29-31 行（包含 `dirichlet_alpha` 的地方）：
   ```python
   cfg.data.dirichlet_alpha = 0.0  # Dirichlet concentration parameter for
   # class heterogeneity. 0.0 = uniform split, 0.1 = highly heterogeneous,
   # 0.5 = moderate heterogeneity, 1.0+ = more balanced
   ```

   在其后添加（注意保持缩进一致）：
   ```python
   cfg.data.server_test_samples_per_class = 10  # Number of samples per class
   # for server-side balanced test sets (used in domain-specific evaluation)

   ```

   **文件 2：`/root/FedLSA/federatedscope/core/configs/cfg_model.py`**

   找到第 16 行：
   ```python
   cfg.model.hidden = 256
   ```

   在其后添加：
   ```python
   cfg.model.num_classes = 10  # Number of classes for classification tasks
   ```

   **文件 3：`/root/FedLSA/federatedscope/core/configs/cfg_fedlsa.py`**

   找到第 76-77 行：
   ```python
   if not hasattr(cfg.model, 'num_classes'):
       cfg.model.num_classes = 10
   ```

   替换为：
   ```python
   # Note: This is handled by the model config, we just document it here
   # cfg.model.num_classes should be set by the model configuration
   ```

3. **验证修复**
   ```bash
   python -c "
   import sys
   sys.path.insert(0, '.')
   from federatedscope.core.configs.config import global_cfg
   cfg = global_cfg.clone()
   print('server_test_samples_per_class:', cfg.data.server_test_samples_per_class)
   print('num_classes:', cfg.model.num_classes)
   "
   ```

### 方案 3：统一代码库路径

如果你想在 `/root/CDA` 工作，可以强制 Python 使用该目录：

```bash
# 在运行前设置环境变量
export PYTHONPATH=/root/CDA:$PYTHONPATH
python federatedscope/main.py --cfg scripts/example_configs/fedavg_office_caltech_domain_eval.yaml
```

或者在配置文件 `~/.bashrc` 中永久设置：
```bash
echo 'export PYTHONPATH=/root/CDA:$PYTHONPATH' >> ~/.bashrc
source ~/.bashrc
```

## 验证修复成功

运行以下命令测试：
```bash
cd /root/CDA  # 或你的工作目录
python federatedscope/main.py --cfg scripts/example_configs/fedavg_office_caltech_domain_eval.yaml
```

如果看到正常的训练日志而不是 `KeyError`，说明修复成功！

## 文件对照清单

修改前后对照：

### cfg_data.py (第 29-34 行)
```python
# 修改前
cfg.data.dirichlet_alpha = 0.0  # ...

cfg.data.transform = [

# 修改后
cfg.data.dirichlet_alpha = 0.0  # ...
cfg.data.server_test_samples_per_class = 10  # Number of samples per class
# for server-side balanced test sets (used in domain-specific evaluation)

cfg.data.transform = [
```

### cfg_model.py (第 16-18 行)
```python
# 修改前
cfg.model.hidden = 256
cfg.model.dropout = 0.5

# 修改后
cfg.model.hidden = 256
cfg.model.num_classes = 10  # Number of classes for classification tasks
cfg.model.dropout = 0.5
```

### cfg_fedlsa.py (第 72-77 行)
```python
# 修改前
# ---------------------------------------------------------------------- #
# FedLSA的模型配置
# ---------------------------------------------------------------------- #
# 类别数量 (论文中的C，用于锚点初始化)
if not hasattr(cfg.model, 'num_classes'):
    cfg.model.num_classes = 10

# 修改后
# ---------------------------------------------------------------------- #
# FedLSA的模型配置
# ---------------------------------------------------------------------- #
# 类别数量 (论文中的C，用于锚点初始化)
# Note: This is handled by the model config, we just document it here
# cfg.model.num_classes should be set by the model configuration
```

## 常见问题

**Q: 我确定修改了文件，为什么还是报错？**

A: 确认你修改的是 Python 实际导入的那个目录。使用 `diagnose_import_path.py` 脚本查看。

**Q: 如何确认修改生效？**

A: 在 Python 中运行：
```python
from federatedscope.core.configs.config import global_cfg
print(hasattr(global_cfg.data, 'server_test_samples_per_class'))  # 应该是 True
```

**Q: 可以直接删除一个代码库吗？**

A: 可以，但要确保：
1. 删除的不是你在用的版本
2. 备份重要的修改
3. 更新 PYTHONPATH 环境变量

## 需要帮助？

如果以上方法都不能解决问题，请提供：
1. `diagnose_import_path.py` 的完整输出
2. 报错的完整 traceback
3. 你修改的文件的具体路径
