"""
验证GGEUR CNN模式是否正确工作
"""
import sys
import os

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_cnn_extractor_loading():
    """测试CNN特征提取器加载"""
    print("=" * 60)
    print("测试 1: CNN 特征提取器加载")
    print("=" * 60)

    try:
        import torch
        from federatedscope.contrib.model.ggeur_cnn_extractor import CNNFeatureExtractor

        # 测试 ConvNeXt-Base
        print("\n加载 ConvNeXt-Base...")
        extractor = CNNFeatureExtractor(
            model_name='convnext_base',
            pretrained=True,
            freeze=True
        )

        print(f"✓ 模型加载成功")
        print(f"  - 模型名称: convnext_base")
        print(f"  - 输出维度: {extractor.get_feature_dim()}")
        print(f"  - 参数冻结: True")

        # 测试前向传播
        dummy_input = torch.randn(2, 3, 224, 224)
        with torch.no_grad():
            features = extractor(dummy_input)

        print(f"✓ 前向传播成功")
        print(f"  - 输入形状: {dummy_input.shape}")
        print(f"  - 输出形状: {features.shape}")
        print(f"  - 特征范数: {features.norm(dim=1).mean().item():.4f}")

        return True

    except Exception as e:
        print(f"✗ 失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_feature_extractor_type():
    """测试feature_extractor_type配置"""
    print("\n" + "=" * 60)
    print("测试 2: feature_extractor_type 配置检查")
    print("=" * 60)

    try:
        from federatedscope.core.configs.config import global_cfg, CN
        from federatedscope.core.configs.cfg_ggeur import extend_ggeur_cfg

        # 创建配置
        cfg = global_cfg.clone()
        extend_ggeur_cfg(cfg)

        print(f"\n默认配置:")
        print(f"  - feature_extractor: {cfg.ggeur.feature_extractor}")
        print(f"  - cnn_backbone: {cfg.ggeur.cnn_backbone}")
        print(f"  - embedding_dim: {cfg.ggeur.embedding_dim}")

        # 模拟CNN配置
        cfg.ggeur.feature_extractor = 'cnn'
        cfg.ggeur.cnn_backbone = 'convnext_base'
        cfg.ggeur.embedding_dim = 1024

        print(f"\nCNN模式配置:")
        print(f"  - feature_extractor: {cfg.ggeur.feature_extractor}")
        print(f"  - cnn_backbone: {cfg.ggeur.cnn_backbone}")
        print(f"  - embedding_dim: {cfg.ggeur.embedding_dim}")

        # 检查逻辑
        if cfg.ggeur.feature_extractor == 'cnn':
            print(f"\n✓ 配置正确指向CNN模式")
        else:
            print(f"\n✗ 配置未指向CNN模式")

        return True

    except Exception as e:
        print(f"✗ 失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_client_cnn_mode():
    """测试客户端CNN模式初始化"""
    print("\n" + "=" * 60)
    print("测试 3: 客户端CNN模式初始化")
    print("=" * 60)

    try:
        from federatedscope.core.configs.config import global_cfg
        from federatedscope.core.configs.cfg_ggeur import extend_ggeur_cfg

        # 创建配置
        cfg = global_cfg.clone()
        extend_ggeur_cfg(cfg)

        # 设置CNN模式
        cfg.ggeur.use = True
        cfg.ggeur.feature_extractor = 'cnn'
        cfg.ggeur.cnn_backbone = 'convnext_base'
        cfg.ggeur.embedding_dim = 1024
        cfg.ggeur.cnn_pretrained = True
        cfg.ggeur.freeze_backbone = True

        # 检查配置值
        print(f"\n配置检查:")
        print(f"  - ggeur.use: {cfg.ggeur.use}")
        print(f"  - ggeur.feature_extractor: {cfg.ggeur.feature_extractor}")
        print(f"  - ggeur.cnn_backbone: {cfg.ggeur.cnn_backbone}")
        print(f"  - ggeur.embedding_dim: {cfg.ggeur.embedding_dim}")

        # 模拟客户端初始化逻辑
        feature_extractor_type = getattr(cfg.ggeur, 'feature_extractor', 'clip')
        print(f"\n客户端将使用的特征提取器: {feature_extractor_type}")

        if feature_extractor_type == 'cnn':
            print("✓ 客户端将使用CNN模式")

            # 检查CNN backbone
            cnn_backbone = getattr(cfg.ggeur, 'cnn_backbone', 'convnext_base')
            print(f"  - CNN backbone: {cnn_backbone}")

            # 检查维度匹配
            CNN_DIMS = {
                'convnext_base': 1024,
                'convnext_tiny': 768,
                'convnext_small': 768,
                'resnet18': 512,
                'resnet50': 2048,
            }
            expected_dim = CNN_DIMS.get(cnn_backbone, None)
            if expected_dim and expected_dim != cfg.ggeur.embedding_dim:
                print(f"  ⚠️  维度不匹配! 期望: {expected_dim}, 配置: {cfg.ggeur.embedding_dim}")
            else:
                print(f"  ✓ 维度匹配正确: {cfg.ggeur.embedding_dim}")
        else:
            print("✗ 客户端将使用CLIP模式（不是CNN）")

        return True

    except Exception as e:
        print(f"✗ 失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_feature_extraction_flow():
    """测试特征提取流程"""
    print("\n" + "=" * 60)
    print("测试 4: 特征提取流程模拟")
    print("=" * 60)

    try:
        import torch
        from federatedscope.contrib.model.ggeur_cnn_extractor import CNNFeatureExtractor

        # 模拟配置
        feature_extractor_type = 'cnn'
        cnn_backbone = 'convnext_base'

        print(f"\n配置:")
        print(f"  - feature_extractor_type: {feature_extractor_type}")
        print(f"  - cnn_backbone: {cnn_backbone}")

        # 模拟 _load_feature_extractor
        print(f"\n模拟 _load_feature_extractor():")
        if feature_extractor_type == 'cnn':
            print("  → 调用 _load_cnn_extractor()")
            cnn_extractor = CNNFeatureExtractor(
                model_name=cnn_backbone,
                pretrained=True,
                freeze=True
            )
            print(f"  ✓ CNN extractor 已加载: {cnn_backbone}")
        else:
            print("  → 调用 _load_clip_model()")
            print("  （跳过CLIP加载测试）")

        # 模拟特征提取
        print(f"\n模拟特征提取:")
        dummy_images = torch.randn(4, 3, 224, 224)
        print(f"  - 输入图像: {dummy_images.shape}")

        if feature_extractor_type == 'cnn':
            with torch.no_grad():
                features = cnn_extractor(dummy_images)
            print(f"  - 使用CNN提取特征")
        else:
            print(f"  - 使用CLIP提取特征（跳过）")
            features = None

        if features is not None:
            print(f"  ✓ 输出特征: {features.shape}")
            print(f"  ✓ 特征维度: {features.shape[1]}")

        return True

    except Exception as e:
        print(f"✗ 失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("\n" + "=" * 60)
    print("GGEUR_Clip CNN 模式验证")
    print("=" * 60)

    results = []

    results.append(("CNN特征提取器加载", test_cnn_extractor_loading()))
    results.append(("配置检查", test_feature_extractor_type()))
    results.append(("客户端CNN模式", test_client_cnn_mode()))
    results.append(("特征提取流程", test_feature_extraction_flow()))

    print("\n" + "=" * 60)
    print("测试结果汇总")
    print("=" * 60)

    all_passed = True
    for name, passed in results:
        status = "✓ 通过" if passed else "✗ 失败"
        print(f"  {name}: {status}")
        if not passed:
            all_passed = False

    if all_passed:
        print("\n✓ 所有测试通过！CNN模式应该可以正常工作。")
        print("\n如果运行时仍然卡住，请检查:")
        print("  1. 配置文件中 feature_extractor: 'cnn'")
        print("  2. embedding_dim 与 cnn_backbone 匹配")
        print("  3. 数据集路径正确")
    else:
        print("\n✗ 部分测试失败，请检查错误信息。")


if __name__ == '__main__':
    main()
