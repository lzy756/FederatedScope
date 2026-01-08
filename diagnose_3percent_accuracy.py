"""
诊断GGEUR训练准确率只有3%的问题
检查：
1. CLIP模型权重是否正确加载
2. CLIP特征提取质量
3. 训练参数设置
4. 数据预处理流程
"""

import sys
import os
import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from federatedscope.core.configs.config import CN, init_global_cfg


def check_clip_model_loading(config_path):
    """检查CLIP模型是否正确加载"""

    print("="*80)
    print("步骤1: 检查CLIP模型加载")
    print("="*80)

    # 加载配置
    cfg = CN()
    init_global_cfg(cfg)
    cfg.merge_from_file(config_path)

    clip_model = cfg.ggeur.clip_model
    clip_custom_weights = cfg.ggeur.clip_custom_weights

    print(f"\n配置中的CLIP模型: {clip_model}")
    print(f"配置中的权重路径: {clip_custom_weights}")

    # 展开路径
    import os.path
    weights_path = os.path.expanduser(clip_custom_weights)
    print(f"展开后的权重路径: {weights_path}")

    # 检查文件是否存在
    if not os.path.exists(weights_path):
        print(f"\n❌ 错误: 权重文件不存在！")
        print(f"   路径: {weights_path}")
        print(f"\n可能的原因:")
        print(f"   1. 文件路径错误")
        print(f"   2. 文件还没有上传到服务器")
        print(f"   3. 文件名拼写错误")
        return False
    else:
        print(f"\n✓ 权重文件存在")

        # 检查文件大小
        file_size = os.path.getsize(weights_path) / (1024 * 1024)  # MB
        print(f"   文件大小: {file_size:.2f} MB")

        if file_size < 10:
            print(f"\n⚠️  警告: 文件太小，可能不是有效的CLIP权重")
            print(f"   ViT-B/16权重文件通常在300-400MB")
        elif file_size > 100 and file_size < 500:
            print(f"   文件大小正常 (ViT-B/16通常约350MB)")

    # 尝试加载CLIP模型
    print(f"\n尝试加载CLIP模型...")

    try:
        from federatedscope.contrib.utils.clip_extractor import CLIPExtractor

        extractor = CLIPExtractor(
            model_name=clip_model,
            custom_weights=clip_custom_weights,
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )

        print(f"\n✓ CLIP模型加载成功!")
        print(f"   模型: {extractor.model_name}")
        print(f"   设备: {extractor.device}")
        print(f"   嵌入维度: {extractor.embedding_dim}")

        # 测试特征提取
        print(f"\n测试特征提取...")
        dummy_images = torch.randn(4, 3, 224, 224)
        features = extractor.extract_features(dummy_images)

        print(f"   输入: {dummy_images.shape}")
        print(f"   输出: {features.shape}")
        print(f"   特征范数: {features.norm(dim=-1).mean():.4f} (应该接近1.0)")

        # 检查特征是否合理
        if features.norm(dim=-1).mean() < 0.5 or features.norm(dim=-1).mean() > 2.0:
            print(f"\n⚠️  警告: 特征范数异常，可能模型未正确初始化")
        else:
            print(f"   ✓ 特征范数正常")

        return True

    except Exception as e:
        print(f"\n❌ CLIP模型加载失败!")
        print(f"   错误: {e}")
        import traceback
        traceback.print_exc()
        return False


def check_training_config(config_path):
    """检查训练配置"""

    print("\n" + "="*80)
    print("步骤2: 检查训练配置")
    print("="*80)

    cfg = CN()
    init_global_cfg(cfg)
    cfg.merge_from_file(config_path)

    print(f"\n训练参数:")
    print(f"   学习率: {cfg.train.optimizer.lr}")
    print(f"   优化器: {cfg.train.optimizer.type}")
    print(f"   权重衰减: {cfg.train.optimizer.weight_decay}")
    print(f"   本地更新步数: {cfg.train.local_update_steps}")
    print(f"   批次大小: {cfg.data.batch_size}")

    print(f"\n联邦学习参数:")
    print(f"   客户端数量: {cfg.federate.client_num}")
    print(f"   采样客户端数: {cfg.federate.sample_client_num}")
    print(f"   总轮次: {cfg.federate.total_round_num}")

    print(f"\nGGEUR参数:")
    print(f"   使用GGEUR: {cfg.ggeur.use}")
    print(f"   n_samples_per_original: {cfg.ggeur.n_samples_per_original}")
    print(f"   m_samples_per_prototype: {cfg.ggeur.m_samples_per_prototype}")
    print(f"   target_size_per_class: {cfg.ggeur.target_size_per_class}")
    print(f"   use_full_covariance: {cfg.ggeur.use_full_covariance}")
    print(f"   eigenvalue_scaling: {cfg.ggeur.eigenvalue_scaling}")

    # 检查是否有问题
    issues = []

    if cfg.train.optimizer.lr > 0.1:
        issues.append("学习率过高 (>0.1)")
    elif cfg.train.optimizer.lr < 0.0001:
        issues.append("学习率过低 (<0.0001)")

    if cfg.train.local_update_steps < 10:
        issues.append("本地更新步数过少 (<10)")

    if not cfg.ggeur.use:
        issues.append("GGEUR未启用！")

    if issues:
        print(f"\n⚠️  发现潜在问题:")
        for issue in issues:
            print(f"   - {issue}")
    else:
        print(f"\n✓ 训练配置看起来正常")


def check_data_distribution(config_path):
    """检查数据分布（LDS）"""

    print("\n" + "="*80)
    print("步骤3: 检查数据分布 (LDS)")
    print("="*80)

    cfg = CN()
    init_global_cfg(cfg)
    cfg.merge_from_file(config_path)

    print(f"\n数据分割器: {cfg.data.splitter}")

    if cfg.data.splitter == 'lda_domain':
        print(f"✓ LDS模式已启用")
        if cfg.data.splitter_args:
            alpha = cfg.data.splitter_args[0].get('alpha', 'N/A')
            print(f"   Alpha: {alpha}")
        else:
            print(f"   ⚠️  警告: 未设置alpha参数")
    else:
        print(f"❌ 错误: 未使用LDS模式!")
        print(f"   当前splitter: {cfg.data.splitter}")
        print(f"   应该设置为: lda_domain")


def main():
    config_path = "scripts/example_configs/ggeur_officehome_lds.yaml"

    if not os.path.exists(config_path):
        print(f"错误: 配置文件不存在: {config_path}")
        return

    print("="*80)
    print("GGEUR_Clip 3%准确率问题诊断")
    print("="*80)
    print(f"\n配置文件: {config_path}\n")

    # 步骤1: 检查CLIP模型
    clip_ok = check_clip_model_loading(config_path)

    # 步骤2: 检查训练配置
    check_training_config(config_path)

    # 步骤3: 检查数据分布
    check_data_distribution(config_path)

    # 总结
    print("\n" + "="*80)
    print("诊断总结")
    print("="*80)

    if not clip_ok:
        print("\n❌ 关键问题: CLIP模型未正确加载")
        print("\n这会导致:")
        print("   - 特征质量极差（随机初始化）")
        print("   - 准确率只有3%左右（随机猜测水平）")
        print("\n解决方案:")
        print("   1. 确保权重文件存在: ~/model/open_clip_vitb16.bin")
        print("   2. 确保权重文件是ViT-B/16的（约350MB）")
        print("   3. 检查上面的错误日志，看具体加载失败的原因")
    else:
        print("\n✓ CLIP模型加载正常")
        print("\n如果准确率仍然只有3%，可能的原因:")
        print("   1. LDS数据分布设置有问题")
        print("   2. 训练参数不合适")
        print("   3. 数据预处理有问题")
        print("\n建议:")
        print("   1. 运行 verify_lds_implementation.py 验证LDS")
        print("   2. 检查训练日志中的损失值")
        print("   3. 对比论文源码的超参数设置")


if __name__ == "__main__":
    main()
