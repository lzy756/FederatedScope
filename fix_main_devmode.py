#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
修复 main.py 的导入路径问题
启用 DEV_MODE 或添加路径设置
"""

import os

def fix_main_py():
    """修复 main.py 的路径设置"""

    main_py_path = 'federatedscope/main.py'

    if not os.path.exists(main_py_path):
        print(f"✗ 错误: 找不到 {main_py_path}")
        print("请在 FederatedScope 根目录下运行此脚本")
        return False

    print("=" * 70)
    print("修复 main.py 的导入路径")
    print("=" * 70)

    # 读取文件
    with open(main_py_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # 检查是否需要修复
    if 'DEV_MODE = True' in content:
        print("\n✓ DEV_MODE 已经是 True，无需修复")
        return True

    if 'sys.path.insert(0, file_dir)' in content and 'DEV_MODE = False' not in content:
        print("\n✓ 路径设置已存在，无需修复")
        return True

    # 备份原文件
    backup_path = main_py_path + '.backup'
    with open(backup_path, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"\n✓ 已备份原文件: {backup_path}")

    # 方案1: 启用 DEV_MODE
    print("\n修复方案: 启用 DEV_MODE")

    modified_content = content.replace(
        'DEV_MODE = False',
        'DEV_MODE = True'
    )

    # 写入修改
    with open(main_py_path, 'w', encoding='utf-8') as f:
        f.write(modified_content)

    print(f"✓ 已修改: {main_py_path}")
    print("  DEV_MODE = False -> DEV_MODE = True")

    # 验证
    print("\n验证修改...")
    with open(main_py_path, 'r', encoding='utf-8') as f:
        new_content = f.read()

    if 'DEV_MODE = True' in new_content:
        print("✓ 修改成功！")
        print("\n现在可以直接运行:")
        print("  python federatedscope/main.py --cfg scripts/example_configs/fedavg_office_caltech_domain_eval.yaml")
        return True
    else:
        print("✗ 修改失败")
        return False


if __name__ == '__main__':
    import sys

    current_dir = os.getcwd()
    print(f"当前目录: {current_dir}\n")

    success = fix_main_py()
    sys.exit(0 if success else 1)
