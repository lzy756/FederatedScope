#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
FederatedScope 启动器 - 确保使用正确的代码路径
解决多个代码库导致的导入路径问题
"""

import sys
import os

# 获取脚本所在目录
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# 强制将当前目录放在 sys.path 的最前面
if SCRIPT_DIR in sys.path:
    sys.path.remove(SCRIPT_DIR)
sys.path.insert(0, SCRIPT_DIR)

# 清除所有已导入的 federatedscope 模块
modules_to_remove = [key for key in list(sys.modules.keys()) if key.startswith('federatedscope')]
for module in modules_to_remove:
    del sys.modules[module]

print("=" * 70)
print("FederatedScope 启动器")
print("=" * 70)
print(f"使用目录: {SCRIPT_DIR}")

# 验证导入路径
try:
    import federatedscope
    actual_path = federatedscope.__file__
    module_dir = os.path.dirname(os.path.dirname(actual_path))

    print(f"federatedscope 导入自: {module_dir}")

    if module_dir == SCRIPT_DIR:
        print("✓ 导入路径正确！")
    else:
        print("✗ 导入路径错误！")
        print(f"  期望: {SCRIPT_DIR}")
        print(f"  实际: {module_dir}")
        print("\n可能的原因:")
        print("  1. federatedscope 被安装为系统包")
        print("  2. PYTHONPATH 环境变量指向了其他目录")
        print("\n解决方案:")
        print("  方式1: 卸载系统安装的 federatedscope")
        print("    pip uninstall federatedscope")
        print("  方式2: 使用绝对路径运行")
        print(f"    cd {SCRIPT_DIR} && python -m federatedscope.main --cfg ...")
        sys.exit(1)

except ImportError as e:
    print(f"✗ 无法导入 federatedscope: {e}")
    sys.exit(1)

print("=" * 70)
print()

# 运行主程序
if __name__ == '__main__':
    # 修改当前工作目录
    os.chdir(SCRIPT_DIR)

    # 导入并运行 main
    from federatedscope import main

    # main.py 会从 sys.argv 读取参数
    # 我们不需要修改 sys.argv，因为它已经包含了命令行参数
