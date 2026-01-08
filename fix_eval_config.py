#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
修复 fedavg_domain_eval.py 中的配置访问问题
将 cfg.data.batch_size 改为 cfg.dataloader.batch_size
"""

import os

def fix_fedavg_domain_eval():
    """修复 fedavg_domain_eval.py 的配置访问"""

    file_path = 'federatedscope/contrib/worker/fedavg_domain_eval.py'

    if not os.path.exists(file_path):
        print(f"✗ 错误: 找不到 {file_path}")
        return False

    print("=" * 70)
    print("修复 fedavg_domain_eval.py 的配置访问")
    print("=" * 70)

    # 读取文件
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # 检查是否需要修复
    if 'self._cfg.dataloader.batch_size' in content and 'self._cfg.data.batch_size' not in content:
        print("\n✓ 配置访问已经正确，无需修复")
        return True

    # 备份原文件
    backup_path = file_path + '.backup'
    with open(backup_path, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"\n✓ 已备份原文件: {backup_path}")

    # 修复配置访问
    print("\n修复内容:")
    modifications = []

    # 1. 修复 batch_size 访问
    if 'self._cfg.data.batch_size' in content:
        content = content.replace(
            'self._cfg.data.batch_size',
            'self._cfg.dataloader.batch_size'
        )
        modifications.append("  ✓ self._cfg.data.batch_size -> self._cfg.dataloader.batch_size")

    # 2. 修复 num_workers 访问
    if 'self._cfg.data.num_workers' in content:
        content = content.replace(
            'self._cfg.data.num_workers',
            'self._cfg.dataloader.num_workers'
        )
        modifications.append("  ✓ self._cfg.data.num_workers -> self._cfg.dataloader.num_workers")

    if modifications:
        for mod in modifications:
            print(mod)

        # 写入修改
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)

        print(f"\n✓ 已修改: {file_path}")
        return True
    else:
        print("  无需修改")
        return True


def fix_eval_config():
    """修复评估配置，使用更通用的键"""

    config_file = 'scripts/example_configs/fedavg_office_caltech_domain_eval.yaml'

    if not os.path.exists(config_file):
        print(f"\n✗ 警告: 找不到 {config_file}")
        return False

    print("\n" + "=" * 70)
    print("检查评估配置")
    print("=" * 70)

    with open(config_file, 'r', encoding='utf-8') as f:
        content = f.read()

    # 检查配置
    if 'best_res_update_round_wise_key:' in content:
        if "best_res_update_round_wise_key: 'test_acc'" in content or \
           'best_res_update_round_wise_key: "test_acc"' in content:
            print("\n当前配置: best_res_update_round_wise_key: 'test_acc'")
            print("\n说明: 该配置要求评估结果中必须有 'test_acc' 键")
            print("如果评估失败，可以尝试:")
            print("  1. 设置为 'val_loss' (使用验证集损失)")
            print("  2. 设置为 '' (禁用最佳结果更新)")
            print("\n当前配置暂时保持不变")
        else:
            print(f"\n✓ best_res_update_round_wise_key 配置: {content}")

    return True


if __name__ == '__main__':
    import sys

    current_dir = os.getcwd()
    print(f"当前目录: {current_dir}\n")

    success1 = fix_fedavg_domain_eval()
    success2 = fix_eval_config()

    if success1 and success2:
        print("\n" + "=" * 70)
        print("✓ 修复完成")
        print("=" * 70)
        print("\n现在可以重新运行:")
        print("  python federatedscope/main.py --cfg scripts/example_configs/fedavg_office_caltech_domain_eval.yaml")
        sys.exit(0)
    else:
        print("\n✗ 修复失败")
        sys.exit(1)
