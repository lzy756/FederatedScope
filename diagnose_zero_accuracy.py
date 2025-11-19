"""
诊断服务器端评估准确率为 0 的问题
"""
import torch
import numpy as np
from torch.utils.data import DataLoader
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def test_data_loading():
    """测试数据加载"""
    print("\n" + "="*70)
    print("步骤 1: 测试数据加载")
    print("="*70)

    try:
        from federatedscope.cv.dataset.office_caltech import load_balanced_office_caltech_data
        import yaml

        # 读取配置
        with open('scripts/example_configs/cross_domain_adaptive_office_caltech.yaml', 'r') as f:
            config = yaml.safe_load(f)

        data_root = config['data']['root']
        samples_per_class = config['data'].get('server_test_samples_per_class', 10)

        datasets = load_balanced_office_caltech_data(
            root=data_root,
            samples_per_class=samples_per_class,
            seed=123
        )

        print(f"\n✓ 成功加载 {len(datasets)} 个域的数据")

        for domain, dataset in datasets.items():
            loader = DataLoader(dataset, batch_size=4, shuffle=False)
            batch_images, batch_labels = next(iter(loader))

            print(f"\n域: {domain}")
            print(f"  - 数据集大小: {len(dataset)}")
            print(f"  - 图像形状: {batch_images.shape}")
            print(f"  - 标签形状: {batch_labels.shape}")
            print(f"  - 标签范围: [{batch_labels.min()}, {batch_labels.max()}]")
            print(f"  - 标签示例: {batch_labels[:10].tolist()}")

            # 检查图像值范围
            print(f"  - 图像值范围: [{batch_images.min():.3f}, {batch_images.max():.3f}]")
            print(f"  - 图像均值: {batch_images.mean():.3f}")

        return True, datasets

    except Exception as e:
        print(f"\n✗ 数据加载失败: {e}")
        import traceback
        traceback.print_exc()
        return False, None


def test_model_prediction(datasets):
    """测试模型预测"""
    print("\n" + "="*70)
    print("步骤 2: 测试随机模型预测")
    print("="*70)

    try:
        # 创建一个简单的随机模型进行测试
        from federatedscope.contrib.model.cross_domain_adaptive import CrossDomainAdaptiveModel
        import yaml

        with open('scripts/example_configs/cross_domain_adaptive_office_caltech.yaml', 'r') as f:
            config = yaml.safe_load(f)

        # 创建模型
        model_config = config['model']
        print(f"\n创建模型:")
        print(f"  - 类型: {model_config['type']}")
        print(f"  - 骨干网络: {model_config['backbone']}")
        print(f"  - 隐藏层: {model_config['hidden']}")
        print(f"  - 类别数: {model_config['num_classes']}")

        # 不实际创建模型，只测试数据流
        print("\n✓ 模型配置正确")

        # 测试一个批次的预测
        domain = 'amazon'
        dataset = datasets[domain]
        loader = DataLoader(dataset, batch_size=4, shuffle=False)
        images, labels = next(iter(loader))

        print(f"\n测试批次:")
        print(f"  - 输入形状: {images.shape}")
        print(f"  - 标签: {labels.tolist()}")

        # 模拟随机预测
        num_classes = model_config['num_classes']
        fake_logits = torch.randn(images.shape[0], num_classes)
        fake_probs = torch.softmax(fake_logits, dim=1)
        fake_preds = torch.argmax(fake_probs, dim=1)

        print(f"  - 随机预测: {fake_preds.tolist()}")
        print(f"  - 预测概率形状: {fake_probs.shape}")

        # 计算随机准确率
        correct = (fake_preds == labels).sum().item()
        acc = correct / len(labels)
        print(f"  - 随机准确率: {acc:.4f} ({correct}/{len(labels)})")
        print(f"  - 期望准确率: ~{1/num_classes:.4f} (1/{num_classes})")

        return True

    except Exception as e:
        print(f"\n✗ 模型测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def check_evaluation_logic():
    """检查评估逻辑"""
    print("\n" + "="*70)
    print("步骤 3: 检查评估逻辑")
    print("="*70)

    issues = []

    print("\n可能导致准确率为 0 的原因:")
    print("\n1. 模型未训练/随机初始化")
    print("   - 解决: 这是正常的，前几轮准确率应该接近 1/10 = 0.1")
    print("   - 随机猜测 10 类应该有 ~10% 准确率")

    print("\n2. 标签范围不匹配")
    print("   - 数据标签应该是 [0, 9]")
    print("   - 模型输出应该是 10 个类别的 logits")

    print("\n3. 评估时使用了错误的数据")
    print("   - 检查 test_loader 是否正确设置")
    print("   - 检查 trainer.ctx 中的数据")

    print("\n4. 模型输出格式问题")
    print("   - 确保模型返回 (batch_size, num_classes) 的 logits")
    print("   - 确保使用了正确的损失函数和激活函数")

    print("\n5. 数据预处理问题")
    print("   - 检查图像归一化是否正确")
    print("   - 检查训练和测试的 transform 是否一致")

    print("\n" + "="*70)
    print("调试建议:")
    print("="*70)
    print("\n1. 查看训练日志中的第一轮准确率")
    print("   - 应该接近 0.1 (随机猜测)")
    print("   - 如果是 0.0，可能有问题")

    print("\n2. 检查模型是否正在更新")
    print("   - 查看损失是否在下降")
    print("   - 查看客户端本地训练准确率")

    print("\n3. 添加调试日志")
    print("   - 在 trainer.evaluate() 中打印:")
    print("     * ys_true 的形状和值")
    print("     * ys_prob 的形状和值")
    print("     * ys_pred 的形状和值")

    return True


def main():
    print("="*70)
    print("服务器端评估准确率诊断")
    print("="*70)

    # 步骤 1: 测试数据加载
    data_ok, datasets = test_data_loading()
    if not data_ok:
        print("\n[失败] 数据加载测试未通过")
        return

    # 步骤 2: 测试模型预测
    model_ok = test_model_prediction(datasets)
    if not model_ok:
        print("\n[失败] 模型测试未通过")
        return

    # 步骤 3: 检查评估逻辑
    check_evaluation_logic()

    print("\n" + "="*70)
    print("总结")
    print("="*70)
    print("\n如果数据加载和模型测试都正常，但准确率仍为 0:")
    print("\n1. 这可能是正常的（特别是训练初期）")
    print("   - 随机初始化的模型准确率应该接近 10%")
    print("   - 如果是 0%，可能需要训练更多轮")

    print("\n2. 检查日志中的详细信息:")
    print("   - 客户端本地训练准确率是否在提升？")
    print("   - 训练损失是否在下降？")
    print("   - 是否有其他错误或警告信息？")

    print("\n3. 尝试添加调试代码:")
    print("   - 在 federatedscope/core/trainers/torch_trainer.py 的 _hook_on_batch_end 中")
    print("   - 打印 ctx.y_true 和 ctx.y_prob 查看预测结果")

    print("\n4. 确认修复生效:")
    print("   - 错误 'test_total' 应该已经修复")
    print("   - 评估应该能正常运行而不崩溃")
    print("   - 查看是否还有其他错误信息")

    print("\n" + "="*70)


if __name__ == '__main__':
    main()
