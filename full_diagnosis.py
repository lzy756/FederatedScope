"""
全面诊断GGEUR训练过程
检查每个环节的数据流和数值
"""

import sys
import os
import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main():
    print("="*80)
    print("GGEUR_Clip 训练过程诊断")
    print("="*80)

    print("\n请提供以下信息：")
    print("\n1. 训练日志中的第一轮准确率:")
    print("   例如: Round #1 test_acc=0.03")
    print("\n2. 训练日志中的损失值:")
    print("   例如: train_loss=4.17")
    print("\n3. 查看日志中是否有这一行:")
    print("   '=== TRAINING BATCH INSPECTION ==='")
    print("   如果有，请提供:")
    print("   - Embeddings norm mean: (应该接近1.0)")
    print("   - Labels unique: (应该有多个不同的类别)")
    print("\n4. 查看日志中MLP构建信息:")
    print("   'Built MLP classifier: ...'")
    print("   确认 softmax_output=False")

    print("\n" + "="*80)
    print("关键检查点")
    print("="*80)

    checks = {
        "CLIP权重加载": "查找: 'Successfully loaded custom weights'",
        "LDS启用": "查找: 'Using Label Distribution Skew (LDS)'",
        "MLP配置": "查找: 'softmax_output=False'",
        "训练数据": "查找: 'TRAINING BATCH INSPECTION' 中的 embeddings norm",
        "GGEUR增强": "查找: 'Performing GGEUR_Clip augmentation'",
    }

    for check, instruction in checks.items():
        print(f"\n{check}:")
        print(f"  {instruction}")

    print("\n" + "="*80)
    print("可能的问题和诊断")
    print("="*80)

    problems = [
        {
            "症状": "第一轮准确率就是3%",
            "可能原因": [
                "1. 训练数据有问题（增强后的数据质量差）",
                "2. 模型未收敛（学习率太高或太低）",
                "3. 数据归一化问题（CLIP特征未归一化）",
                "4. 标签错误（增强时标签不匹配）",
            ],
            "诊断方法": [
                "检查训练损失是否下降",
                "检查 embeddings norm mean 是否接近1.0",
                "检查训练准确率是否提升",
                "对比原始数据和增强数据的分布",
            ]
        },
        {
            "症状": "训练损失不下降",
            "可能原因": [
                "1. 学习率设置不对",
                "2. 梯度消失/爆炸",
                "3. 数据质量问题",
            ],
            "诊断方法": [
                "尝试降低学习率（0.01 → 0.001）",
                "检查梯度范数",
                "检查模型输出分布",
            ]
        },
        {
            "症状": "训练准确率高，测试准确率低",
            "可能原因": [
                "1. 过拟合",
                "2. 训练/测试数据分布不一致",
                "3. 测试时未使用CLIP特征",
            ],
            "诊断方法": [
                "添加dropout或正则化",
                "检查测试时是否提取CLIP特征",
                "对比训练和测试的特征分布",
            ]
        }
    ]

    for i, problem in enumerate(problems, 1):
        print(f"\n问题 {i}: {problem['症状']}")
        print(f"\n可能原因:")
        for cause in problem['可能原因']:
            print(f"  {cause}")
        print(f"\n诊断方法:")
        for method in problem['诊断方法']:
            print(f"  {method}")

    print("\n" + "="*80)
    print("立即行动")
    print("="*80)

    print("\n请运行以下命令并提供完整日志的前200行:")
    print("\n```bash")
    print("python run_ggeur.py --cfg scripts/example_configs/ggeur_officehome_lds.yaml 2>&1 | head -200 > ggeur_log.txt")
    print("```")

    print("\n然后检查日志中的以下关键信息:")
    print("1. CLIP模型加载状态")
    print("2. LDS数据分布")
    print("3. 训练批次的embedding统计")
    print("4. 第一轮的训练/测试准确率和损失")

    print("\n将这些信息发给我，我可以帮你精确定位问题！")


if __name__ == "__main__":
    main()
