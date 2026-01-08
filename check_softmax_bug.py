"""
检查GGEUR训练器的softmax配置问题
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from federatedscope.core.configs.config import CN, init_global_cfg


def check_softmax_config(config_path):
    """检查softmax配置"""

    print("="*80)
    print("检查 use_softmax_output 配置")
    print("="*80)

    cfg = CN()
    init_global_cfg(cfg)
    cfg.merge_from_file(config_path)

    print(f"\nGGEUR配置:")
    print(f"  use_softmax_output: {cfg.ggeur.get('use_softmax_output', 'NOT SET')}")

    if not hasattr(cfg.ggeur, 'use_softmax_output'):
        print(f"\n❌ 错误: use_softmax_output 未在配置中设置！")
        print(f"   trainer会使用默认值 True")
        print(f"   这会导致模型输出softmax概率")
        print(f"   然后CrossEntropyLoss会再做log_softmax")
        print(f"   结果：训练无法收敛！")
        return False
    elif cfg.ggeur.use_softmax_output:
        print(f"\n❌ 错误: use_softmax_output = True")
        print(f"   这会导致softmax + CrossEntropyLoss的双重问题")
        return False
    else:
        print(f"\n✓ use_softmax_output = False (正确)")
        return True


def check_loss_computation():
    """演示softmax + CrossEntropyLoss的问题"""

    import torch
    import torch.nn as nn

    print("\n" + "="*80)
    print("演示 softmax + CrossEntropyLoss 的问题")
    print("="*80)

    # 模拟数据
    batch_size = 4
    num_classes = 65
    logits = torch.randn(batch_size, num_classes)
    labels = torch.randint(0, num_classes, (batch_size,))

    # 正确的做法
    criterion = nn.CrossEntropyLoss()
    loss_correct = criterion(logits, labels)

    # 错误的做法（先softmax，再CrossEntropyLoss）
    probs = torch.softmax(logits, dim=1)
    loss_wrong = criterion(probs, labels)

    print(f"\n正确做法（直接用logits）:")
    print(f"  Loss: {loss_correct.item():.4f}")

    print(f"\n错误做法（先softmax）:")
    print(f"  Loss: {loss_wrong.item():.4f}")

    print(f"\n差异:")
    print(f"  {abs(loss_correct - loss_wrong).item():.4f}")

    if abs(loss_correct - loss_wrong).item() > 0.1:
        print(f"\n⚠️  损失值差异很大！")
        print(f"   使用softmax + CrossEntropyLoss会导致训练问题")


def main():
    config_path = "scripts/example_configs/ggeur_officehome_lds.yaml"

    if not os.path.exists(config_path):
        print(f"错误: 配置文件不存在: {config_path}")
        return

    # 检查配置
    config_ok = check_softmax_config(config_path)

    # 演示问题
    check_loss_computation()

    # 总结
    print("\n" + "="*80)
    print("总结")
    print("="*80)

    if not config_ok:
        print("\n❌ 发现配置问题！")
        print("\n这解释了为什么准确率只有3%：")
        print("  1. 模型输出softmax概率")
        print("  2. CrossEntropyLoss期望logits")
        print("  3. log(softmax(x)) != logits")
        print("  4. 梯度错误 → 训练无法收敛")
        print("\n解决方案:")
        print("  在配置文件中确保 use_softmax_output: false")
    else:
        print("\n✓ 配置正确")
        print("\n如果准确率仍然很低，问题可能在其他地方:")
        print("  1. 数据增强质量")
        print("  2. 训练参数")
        print("  3. 评估逻辑")


if __name__ == "__main__":
    main()
