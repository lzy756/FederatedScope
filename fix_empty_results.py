#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
修复 fedavg_domain_eval.py 的空结果问题
在调用 update_best_result 前检查结果是否为空
"""

import os
import sys

def fix_fedavg_domain_eval():
    """修复 fedavg_domain_eval.py 中的空结果问题"""

    file_path = 'federatedscope/contrib/worker/fedavg_domain_eval.py'

    if not os.path.exists(file_path):
        print(f"✗ 错误: 找不到 {file_path}")
        return False

    print("=" * 70)
    print("修复 fedavg_domain_eval.py 的空结果问题")
    print("=" * 70)

    # 读取文件
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # 检查是否已经修复
    if "if formatted_eval_res.get('Results_raw'):" in content:
        print("\n✓ 代码已经修复，无需重复修复")
        return True

    # 备份原文件
    backup_path = file_path + '.backup_empty_results'
    with open(backup_path, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"\n✓ 已备份原文件: {backup_path}")

    # 修复第一处：domain-specific evaluation
    old_code_1 = """                # Update best results with domain tag
                self._monitor.update_best_result(
                    self.best_results,
                    formatted_eval_res['Results_raw'],
                    results_type=f"server_global_eval_{domain}"
                )"""

    new_code_1 = """                # Update best results with domain tag (only if results are not empty)
                if formatted_eval_res.get('Results_raw'):
                    self._monitor.update_best_result(
                        self.best_results,
                        formatted_eval_res['Results_raw'],
                        results_type=f"server_global_eval_{domain}"
                    )
                else:
                    logger.warning(f"Empty evaluation results for domain {domain}, skipping best result update")"""

    if old_code_1 in content:
        content = content.replace(old_code_1, new_code_1)
        print("\n✓ 已修复第一处: domain-specific evaluation")
    else:
        print("\n⚠ 警告: 未找到第一处需要修复的代码")

    # 修复第二处：weighted average evaluation
    old_code_2 = """        # Update best results for weighted average
        self._monitor.update_best_result(
            self.best_results,
            formatted_weighted_res['Results_raw'],
            results_type="server_global_eval_weighted"
        )"""

    new_code_2 = """        # Update best results for weighted average (only if results are not empty)
        if formatted_weighted_res.get('Results_raw'):
            self._monitor.update_best_result(
                self.best_results,
                formatted_weighted_res['Results_raw'],
                results_type="server_global_eval_weighted"
            )
        else:
            logger.warning("Empty weighted evaluation results, skipping best result update")"""

    if old_code_2 in content:
        content = content.replace(old_code_2, new_code_2)
        print("✓ 已修复第二处: weighted average evaluation")
    else:
        print("⚠ 警告: 未找到第二处需要修复的代码")

    # 写入修改
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)

    print(f"\n✓ 已修改: {file_path}")
    return True


def main():
    """主函数"""

    current_dir = os.getcwd()
    print(f"当前目录: {current_dir}\n")

    if not os.path.exists('federatedscope/contrib/worker'):
        print("✗ 错误: 当前目录不是 FederatedScope 根目录")
        print("请在 FederatedScope 根目录下运行此脚本:")
        print("  cd /root/CDA_new")
        print("  python fix_empty_results.py")
        return 1

    success = fix_fedavg_domain_eval()

    if success:
        print("\n" + "=" * 70)
        print("✓✓✓ 修复完成！✓✓✓")
        print("=" * 70)
        print("\n修复内容:")
        print("  1. 在 domain-specific evaluation 中添加了空结果检查")
        print("  2. 在 weighted average evaluation 中添加了空结果检查")
        print("\n现在可以运行:")
        print("  python federatedscope/main.py --cfg scripts/example_configs/fedavg_office_caltech_domain_eval.yaml")
        return 0
    else:
        print("\n✗ 修复失败")
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
