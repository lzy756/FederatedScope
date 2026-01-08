#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
FederatedScope 配置修复脚本 - 专门用于 /root/CDA_new
一键修复所有缺失的配置项
"""

import os
import sys
import shutil
from datetime import datetime


def print_section(title):
    """打印分隔线标题"""
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def backup_files(base_dir):
    """备份配置文件"""
    print("\n[备份] 备份原始配置文件...")
    backup_dir = os.path.join(base_dir, f'.config_backup_{datetime.now().strftime("%Y%m%d_%H%M%S")}')
    os.makedirs(backup_dir, exist_ok=True)

    config_files = [
        'federatedscope/core/configs/cfg_data.py',
        'federatedscope/core/configs/cfg_model.py',
        'federatedscope/core/configs/cfg_fedlsa.py',
    ]

    for file_path in config_files:
        full_path = os.path.join(base_dir, file_path)
        if os.path.exists(full_path):
            backup_path = os.path.join(backup_dir, os.path.basename(file_path))
            shutil.copy2(full_path, backup_path)
            print(f"  ✓ 已备份: {os.path.basename(file_path)}")

    print(f"  备份目录: {backup_dir}")
    return backup_dir


def fix_cfg_data(file_path):
    """修复 cfg_data.py"""
    print("\n[1/3] 修复 cfg_data.py ...")

    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    if 'server_test_samples_per_class' in content:
        print("  ✓ server_test_samples_per_class 已存在，无需修复")
        return False

    # 找到插入位置并添加配置
    lines = content.split('\n')
    new_lines = []
    inserted = False

    for i, line in enumerate(lines):
        new_lines.append(line)
        # 在 dirichlet_alpha 的最后一行注释后插入
        if 'cfg.data.dirichlet_alpha' in line and '1.0+ = more balanced' in line:
            new_lines.append('    cfg.data.server_test_samples_per_class = 10  # Number of samples per class')
            new_lines.append('    # for server-side balanced test sets (used in domain-specific evaluation)')
            new_lines.append('')
            inserted = True

    if inserted:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(new_lines))
        print("  ✓ 已添加 server_test_samples_per_class 配置")
        return True
    else:
        print("  ✗ 警告: 未找到插入位置，请手动检查")
        return False


def fix_cfg_model(file_path):
    """修复 cfg_model.py"""
    print("\n[2/3] 修复 cfg_model.py ...")

    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    if 'cfg.model.num_classes' in content:
        print("  ✓ num_classes 已存在，无需修复")
        return False

    # 在 hidden 后添加 num_classes
    lines = content.split('\n')
    new_lines = []
    inserted = False

    for line in lines:
        new_lines.append(line)
        if 'cfg.model.hidden = 256' in line:
            new_lines.append('    cfg.model.num_classes = 10  # Number of classes for classification tasks')
            inserted = True

    if inserted:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(new_lines))
        print("  ✓ 已添加 num_classes 配置")
        return True
    else:
        print("  ✗ 警告: 未找到插入位置，请手动检查")
        return False


def fix_cfg_fedlsa(file_path):
    """修复 cfg_fedlsa.py - 移除错误的 hasattr 代码"""
    print("\n[3/3] 修复 cfg_fedlsa.py ...")

    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    if 'hasattr(cfg.model' not in content:
        print("  ✓ 没有 hasattr 错误，无需修复")
        return False

    print("  需要移除错误的 hasattr 代码...")

    # 移除 hasattr 相关代码
    lines = content.split('\n')
    new_lines = []
    skip_next = False
    fixed = False

    for line in lines:
        if 'if not hasattr(cfg.model' in line:
            skip_next = True
            fixed = True
            # 添加修复后的注释
            new_lines.append('    # Note: This is handled by the model config, we just document it here')
            new_lines.append('    # cfg.model.num_classes should be set by the model configuration')
            continue

        if skip_next and 'cfg.model.num_classes = 10' in line:
            skip_next = False
            continue

        new_lines.append(line)

    if fixed:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(new_lines))
        print("  ✓ 已移除 hasattr 错误代码并添加说明")
        return True
    else:
        print("  ✗ 警告: 未找到需要修复的代码")
        return False


def verify_config(base_dir):
    """验证配置是否正确"""
    print_section("验证修复结果")

    # 切换到目标目录
    original_dir = os.getcwd()
    os.chdir(base_dir)

    # 将目录添加到 Python 路径
    if base_dir not in sys.path:
        sys.path.insert(0, base_dir)

    try:
        # 清除已导入的模块缓存
        modules_to_remove = [key for key in sys.modules.keys() if key.startswith('federatedscope')]
        for module in modules_to_remove:
            del sys.modules[module]

        # 导入配置
        from federatedscope.core.configs.config import global_cfg
        cfg = global_cfg.clone()

        # 检查配置项
        has_server_test = hasattr(cfg.data, 'server_test_samples_per_class')
        has_num_classes = hasattr(cfg.model, 'num_classes')

        print(f"\n配置项检查:")
        print(f"  {'✓' if has_server_test else '✗'} cfg.data.server_test_samples_per_class: {has_server_test}")
        if has_server_test:
            print(f"      默认值 = {cfg.data.server_test_samples_per_class}")

        print(f"  {'✓' if has_num_classes else '✗'} cfg.model.num_classes: {has_num_classes}")
        if has_num_classes:
            print(f"      默认值 = {cfg.model.num_classes}")

        # 测试加载配置文件
        config_file = 'scripts/example_configs/fedavg_office_caltech_domain_eval.yaml'
        if os.path.exists(config_file):
            print(f"\n测试加载配置文件: {config_file}")
            try:
                test_cfg = global_cfg.clone()
                test_cfg.merge_from_file(config_file)
                print(f"  ✓ 配置文件加载成功")
                print(f"      data.server_test_samples_per_class = {test_cfg.data.server_test_samples_per_class}")
                print(f"      model.num_classes = {test_cfg.model.num_classes}")
            except Exception as e:
                print(f"  ✗ 配置文件加载失败: {e}")
                return False

        success = has_server_test and has_num_classes

        os.chdir(original_dir)
        return success

    except Exception as e:
        print(f"\n✗ 验证时出错: {e}")
        import traceback
        traceback.print_exc()
        os.chdir(original_dir)
        return False


def main():
    """主函数"""
    print_section("FederatedScope 配置修复工具 - CDA_new")

    # 检测当前目录
    current_dir = os.getcwd()
    print(f"\n当前工作目录: {current_dir}")

    # 检查是否在正确的目录
    if not os.path.exists('federatedscope/core/configs'):
        print("\n✗ 错误: 当前目录不是 FederatedScope 根目录")
        print("请在 FederatedScope 根目录下运行此脚本:")
        print("  cd /root/CDA_new")
        print("  python fix_cda_new_config.py")
        return 1

    # 确认修复
    print("\n将修复以下配置文件:")
    print("  1. federatedscope/core/configs/cfg_data.py")
    print("  2. federatedscope/core/configs/cfg_model.py")
    print("  3. federatedscope/core/configs/cfg_fedlsa.py")

    response = input("\n是否继续? (y/n): ").strip().lower()
    if response != 'y':
        print("已取消修复")
        return 0

    # 备份文件
    backup_dir = backup_files(current_dir)

    # 修复配置文件
    print_section("开始修复配置文件")

    config_files = {
        'cfg_data.py': os.path.join(current_dir, 'federatedscope/core/configs/cfg_data.py'),
        'cfg_model.py': os.path.join(current_dir, 'federatedscope/core/configs/cfg_model.py'),
        'cfg_fedlsa.py': os.path.join(current_dir, 'federatedscope/core/configs/cfg_fedlsa.py'),
    }

    modified = []

    if fix_cfg_data(config_files['cfg_data.py']):
        modified.append('cfg_data.py')

    if fix_cfg_model(config_files['cfg_model.py']):
        modified.append('cfg_model.py')

    if fix_cfg_fedlsa(config_files['cfg_fedlsa.py']):
        modified.append('cfg_fedlsa.py')

    # 验证修复
    if verify_config(current_dir):
        print_section("✓✓✓ 修复成功！✓✓✓")
        print(f"\n已修改的文件: {', '.join(modified) if modified else '无（已经正确）'}")
        print(f"备份目录: {backup_dir}")
        print("\n现在可以运行:")
        print("  python federatedscope/main.py --cfg scripts/example_configs/fedavg_office_caltech_domain_eval.yaml")
        return 0
    else:
        print_section("✗ 验证失败")
        print(f"\n可以从备份恢复:")
        print(f"  cp {backup_dir}/* federatedscope/core/configs/")
        return 1


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
