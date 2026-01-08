"""
诊断CNN模式GGEUR的问题
"""
import sys
import numpy as np

# 检查配置中的embedding_dim是否与CNN backbone匹配
CNN_FEATURE_DIMS = {
    'convnext_tiny': 768,
    'convnext_small': 768,
    'convnext_base': 1024,
    'convnext_large': 1536,
    'resnet18': 512,
    'resnet34': 512,
    'resnet50': 2048,
    'resnet101': 2048,
    'efficientnet_b0': 1280,
    'efficientnet_b4': 1792,
    'efficientnet_b7': 2560,
}

def check_config(config_path):
    """检查配置文件"""
    import yaml

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    ggeur_cfg = config.get('ggeur', {})

    feature_extractor = ggeur_cfg.get('feature_extractor', 'clip')
    cnn_backbone = ggeur_cfg.get('cnn_backbone', 'convnext_base')
    embedding_dim = ggeur_cfg.get('embedding_dim', 512)

    print("=" * 60)
    print("GGEUR_Clip CNN 配置检查")
    print("=" * 60)
    print(f"Feature Extractor: {feature_extractor}")
    print(f"CNN Backbone: {cnn_backbone}")
    print(f"Configured embedding_dim: {embedding_dim}")

    if feature_extractor == 'cnn':
        expected_dim = CNN_FEATURE_DIMS.get(cnn_backbone, None)
        if expected_dim is None:
            print(f"\n⚠️  警告: 未知的CNN backbone '{cnn_backbone}'")
        elif expected_dim != embedding_dim:
            print(f"\n❌ 错误: embedding_dim 不匹配!")
            print(f"   CNN backbone '{cnn_backbone}' 输出维度: {expected_dim}")
            print(f"   配置的 embedding_dim: {embedding_dim}")
            print(f"\n   请修改配置文件中的 embedding_dim 为 {expected_dim}")
            return False
        else:
            print(f"\n✓ embedding_dim 正确匹配 ({expected_dim})")
    else:
        if embedding_dim != 512:
            print(f"\n⚠️  警告: CLIP模式通常使用 embedding_dim=512")

    return True


def test_cnn_extractor():
    """测试CNN特征提取器"""
    print("\n" + "=" * 60)
    print("测试 CNN 特征提取器")
    print("=" * 60)

    try:
        from federatedscope.contrib.model.ggeur_cnn_extractor import CNNFeatureExtractor
        import torch

        # 测试不同的backbone
        for backbone in ['convnext_base', 'resnet18', 'resnet50']:
            try:
                extractor = CNNFeatureExtractor(
                    model_name=backbone,
                    pretrained=True,
                    freeze=True
                )

                # 测试前向传播
                dummy_input = torch.randn(2, 3, 224, 224)
                with torch.no_grad():
                    features = extractor(dummy_input)

                print(f"✓ {backbone}: 输出维度 = {features.shape[1]}")

            except Exception as e:
                print(f"✗ {backbone}: {e}")

    except ImportError as e:
        print(f"✗ 无法导入 CNNFeatureExtractor: {e}")


def test_augmentation_dimensions():
    """测试增强时的维度匹配"""
    print("\n" + "=" * 60)
    print("测试高斯增强维度")
    print("=" * 60)

    # 模拟CNN特征（1024维）
    feature_dim = 1024
    num_samples = 10

    # 模拟特征
    features = np.random.randn(num_samples, feature_dim)
    mean = np.mean(features, axis=0)

    # 计算协方差
    centered = features - mean
    cov = (1.0 / num_samples) * np.dot(centered.T, centered)

    print(f"特征维度: {feature_dim}")
    print(f"均值形状: {mean.shape}")
    print(f"协方差形状: {cov.shape}")

    # 测试采样
    try:
        # 添加正则化
        cov_reg = cov + 1e-6 * np.eye(feature_dim)

        # Cholesky分解
        L = np.linalg.cholesky(cov_reg)

        # 生成样本
        generated = np.random.multivariate_normal(mean, L @ L.T, 5)
        print(f"✓ 生成样本形状: {generated.shape}")

    except Exception as e:
        print(f"✗ 采样失败: {e}")


def check_mismatch_scenario():
    """检查维度不匹配场景"""
    print("\n" + "=" * 60)
    print("模拟维度不匹配场景（这是可能导致卡住的原因）")
    print("=" * 60)

    # 场景：CNN输出1024维，但embedding_dim配置为512
    cnn_feature_dim = 1024  # CNN实际输出
    config_embedding_dim = 512  # 错误配置

    print(f"CNN实际输出维度: {cnn_feature_dim}")
    print(f"配置的embedding_dim: {config_embedding_dim}")

    # 模拟特征
    features = np.random.randn(10, cnn_feature_dim)
    mean = np.mean(features, axis=0)  # shape: (1024,)

    # 当没有全局协方差时，会创建一个错误维度的单位矩阵
    wrong_cov = np.eye(config_embedding_dim) * 0.01  # shape: (512, 512)

    print(f"\n均值形状: {mean.shape}")
    print(f"协方差形状（错误）: {wrong_cov.shape}")

    try:
        # 这会失败或产生错误结果
        generated = np.random.multivariate_normal(mean, wrong_cov, 5)
        print(f"生成样本形状: {generated.shape}")
    except Exception as e:
        print(f"\n❌ 采样失败（预期）: {type(e).__name__}")
        print(f"   这就是导致程序卡住的原因！")
        print(f"\n   解决方案：将配置中的 embedding_dim 改为 {cnn_feature_dim}")


if __name__ == '__main__':
    if len(sys.argv) > 1:
        config_path = sys.argv[1]
        check_config(config_path)
    else:
        print("用法: python diagnose_cnn_ggeur.py <config.yaml>")
        print("\n运行默认测试...\n")

    test_cnn_extractor()
    test_augmentation_dimensions()
    check_mismatch_scenario()
