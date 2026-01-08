#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
FederatedScope 修复并运行工具
一键修复配置并运行训练
"""

import sys
import os

def main():
    print("=" * 70)
    print("FederatedScope 一键修复并运行")
    print("=" * 70)

    # 1. 确定当前目录
    current_dir = os.getcwd()
    print(f"\n当前目录: {current_dir}")

    # 2. 检查是否在 FederatedScope 目录
    if not os.path.exists('federatedscope/main.py'):
        print("\n✗ 错误: 当前目录不是 FederatedScope 根目录")
        print("请在 FederatedScope 根目录下运行此脚本:")
        print("  cd /root/CDA_new")
        print("  python fix_and_run.py")
        return 1

    # 3. 强制设置 Python 路径
    if current_dir in sys.path:
        sys.path.remove(current_dir)
    sys.path.insert(0, current_dir)

    # 清除已导入的模块
    modules_to_remove = [key for key in list(sys.modules.keys())
                        if key.startswith('federatedscope')]
    for module in modules_to_remove:
        del sys.modules[module]

    print("\n检查导入路径...")

    # 4. 验证导入路径
    try:
        import federatedscope
        actual_path = federatedscope.__file__
        module_dir = os.path.dirname(os.path.dirname(actual_path))

        print(f"  federatedscope 导入自: {module_dir}")

        if module_dir != current_dir:
            print(f"\n✗ 错误: Python 仍在使用其他目录的代码")
            print(f"  当前目录: {current_dir}")
            print(f"  实际导入: {module_dir}")
            print("\n原因: federatedscope 可能被安装为系统包")
            print("\n解决方案:")
            print("=" * 70)
            print("在服务器上运行以下命令:")
            print()
            print("# 方案 1: 卸载系统安装的 federatedscope (推荐)")
            print("pip uninstall federatedscope -y")
            print()
            print("# 方案 2: 或者使用以下方式运行")
            print(f"cd {current_dir}")
            print("PYTHONPATH=. python federatedscope/main.py --cfg scripts/example_configs/fedavg_office_caltech_domain_eval.yaml")
            print()
            print("# 方案 3: 或者修改环境变量")
            print(f"export PYTHONPATH={current_dir}:$PYTHONPATH")
            print("python federatedscope/main.py --cfg scripts/example_configs/fedavg_office_caltech_domain_eval.yaml")
            print("=" * 70)
            return 1
        else:
            print("  ✓ 导入路径正确")

    except ImportError as e:
        print(f"  ✗ 无法导入: {e}")
        return 1

    # 5. 检查配置
    print("\n检查配置...")
    try:
        from federatedscope.core.configs.config import global_cfg
        cfg = global_cfg.clone()

        has_server_test = hasattr(cfg.data, 'server_test_samples_per_class')
        has_num_classes = hasattr(cfg.model, 'num_classes')

        print(f"  {'✓' if has_server_test else '✗'} server_test_samples_per_class: {has_server_test}")
        print(f"  {'✓' if has_num_classes else '✗'} num_classes: {has_num_classes}")

        if not (has_server_test and has_num_classes):
            print("\n✗ 配置不完整，请先运行 fix_cda_new_config.py")
            return 1

    except Exception as e:
        print(f"  ✗ 配置检查失败: {e}")
        return 1

    # 6. 测试加载配置文件
    print("\n测试加载配置文件...")
    config_file = 'scripts/example_configs/fedavg_office_caltech_domain_eval.yaml'

    if not os.path.exists(config_file):
        print(f"  ✗ 配置文件不存在: {config_file}")
        return 1

    try:
        test_cfg = global_cfg.clone()
        test_cfg.merge_from_file(config_file)
        print(f"  ✓ 配置文件加载成功")
    except Exception as e:
        print(f"  ✗ 配置文件加载失败: {e}")
        import traceback
        traceback.print_exc()
        return 1

    print("\n" + "=" * 70)
    print("✓ 所有检查通过！")
    print("=" * 70)

    # 7. 询问是否运行
    response = input("\n是否开始运行训练? (y/n): ").strip().lower()
    if response != 'y':
        print("已取消运行")
        print("\n你可以手动运行:")
        print(f"  cd {current_dir}")
        print(f"  python federatedscope/main.py --cfg {config_file}")
        return 0

    # 8. 运行训练
    print("\n" + "=" * 70)
    print("开始训练...")
    print("=" * 70)
    print()

    # 修改 sys.argv 来传递参数
    sys.argv = [
        'federatedscope/main.py',
        '--cfg', config_file
    ]

    # 执行 main.py
    exec(open('federatedscope/main.py').read())

    return 0


if __name__ == '__main__':
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\n已取消")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ 错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
