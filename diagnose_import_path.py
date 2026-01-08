#!/usr/bin/env python
"""
诊断脚本：检查 Python 实际导入的 federatedscope 模块路径
帮助定位多个代码库导致的路径问题
"""

import sys
import os

print("=" * 70)
print("FederatedScope 导入路径诊断工具")
print("=" * 70)

# 1. 显示当前工作目录
print(f"\n[1] 当前工作目录:")
print(f"    {os.getcwd()}")

# 2. 显示 sys.path
print(f"\n[2] Python 模块搜索路径 (sys.path):")
for i, path in enumerate(sys.path, 1):
    print(f"    {i}. {path}")

# 3. 尝试导入 federatedscope 并显示实际路径
print(f"\n[3] 导入 federatedscope 模块...")
try:
    import federatedscope
    actual_path = federatedscope.__file__
    module_dir = os.path.dirname(os.path.dirname(actual_path))
    print(f"    ✓ 成功导入")
    print(f"    模块文件: {actual_path}")
    print(f"    模块目录: {module_dir}")
except ImportError as e:
    print(f"    ✗ 导入失败: {e}")
    sys.exit(1)

# 4. 检查配置文件路径
print(f"\n[4] 检查关键配置文件:")
config_files = [
    'federatedscope/core/configs/cfg_data.py',
    'federatedscope/core/configs/cfg_model.py',
    'federatedscope/core/configs/cfg_fedlsa.py',
]

for config_file in config_files:
    full_path = os.path.join(module_dir, config_file)
    exists = os.path.exists(full_path)
    status = "✓" if exists else "✗"
    print(f"    {status} {full_path}")

    # 检查是否包含必要的配置
    if exists and 'cfg_data.py' in config_file:
        with open(full_path, 'r', encoding='utf-8') as f:
            content = f.read()
            has_config = 'server_test_samples_per_class' in content
            print(f"       {'✓' if has_config else '✗'} 包含 server_test_samples_per_class: {has_config}")

    if exists and 'cfg_model.py' in config_file:
        with open(full_path, 'r', encoding='utf-8') as f:
            content = f.read()
            has_config = 'cfg.model.num_classes' in content
            print(f"       {'✓' if has_config else '✗'} 包含 cfg.model.num_classes: {has_config}")

    if exists and 'cfg_fedlsa.py' in config_file:
        with open(full_path, 'r', encoding='utf-8') as f:
            content = f.read()
            has_error = 'hasattr(cfg.model' in content
            print(f"       {'✗' if has_error else '✓'} {'包含错误的 hasattr 代码' if has_error else '无 hasattr 错误'}")

# 5. 检查是否有多个 FederatedScope 目录
print(f"\n[5] 搜索可能的 FederatedScope 安装位置:")
potential_dirs = [
    '/root/CDA',
    '/root/FedLSA',
    '/root/FederatedScope',
    os.path.expanduser('~/CDA'),
    os.path.expanduser('~/FedLSA'),
]

found_dirs = []
for dir_path in potential_dirs:
    fs_path = os.path.join(dir_path, 'federatedscope')
    if os.path.isdir(fs_path):
        found_dirs.append(dir_path)
        is_active = (module_dir == dir_path)
        status = ">>> 当前使用 <<<" if is_active else "   未使用"
        print(f"    {status} {dir_path}")

# 6. 环境变量检查
print(f"\n[6] 环境变量检查:")
pythonpath = os.environ.get('PYTHONPATH', '')
if pythonpath:
    print(f"    PYTHONPATH = {pythonpath}")
else:
    print(f"    PYTHONPATH 未设置")

# 7. 给出建议
print(f"\n{'=' * 70}")
print("诊断结果和建议:")
print("=" * 70)

if len(found_dirs) > 1:
    print(f"\n⚠️  发现多个 FederatedScope 安装位置:")
    for d in found_dirs:
        print(f"   - {d}")
    print(f"\n当前 Python 使用的是: {module_dir}")
    print(f"\n建议:")
    print(f"  1. 确认你想使用哪个版本的代码库")
    print(f"  2. 在那个目录下修改配置文件")
    print(f"  3. 或者在运行前设置 PYTHONPATH:")
    print(f"     export PYTHONPATH=/path/to/correct/FederatedScope:$PYTHONPATH")
    print(f"  4. 或者删除/移除不需要的代码库")
else:
    print(f"\n✓ 只找到一个 FederatedScope 安装")
    print(f"  位置: {module_dir}")

# 8. 测试配置加载
print(f"\n[8] 测试配置加载:")
try:
    from federatedscope.core.configs.config import global_cfg
    print(f"    ✓ 成功导入 global_cfg")

    # 检查配置是否包含新增的字段
    cfg = global_cfg.clone()

    has_server_test = hasattr(cfg.data, 'server_test_samples_per_class')
    has_num_classes = hasattr(cfg.model, 'num_classes')

    print(f"    {'✓' if has_server_test else '✗'} cfg.data.server_test_samples_per_class: {has_server_test}")
    if has_server_test:
        print(f"      默认值 = {cfg.data.server_test_samples_per_class}")

    print(f"    {'✓' if has_num_classes else '✗'} cfg.model.num_classes: {has_num_classes}")
    if has_num_classes:
        print(f"      默认值 = {cfg.model.num_classes}")

    if has_server_test and has_num_classes:
        print(f"\n✓ 配置正确！可以正常使用")
    else:
        print(f"\n✗ 配置不完整！需要在以下目录修改:")
        print(f"   {module_dir}/federatedscope/core/configs/")

except Exception as e:
    print(f"    ✗ 加载失败: {e}")
    import traceback
    traceback.print_exc()

print(f"\n{'=' * 70}\n")
