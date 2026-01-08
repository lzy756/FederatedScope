"""
GGEUR_Clip 修复验证脚本

运行这个脚本来验证所有关键修复是否正确应用。
"""

import torch
import sys
import os

# 添加项目路径
sys.path.insert(0, 'D:/Projects/FederatedScope')

def test_eigenvalue_order():
    """测试特征值是否保持升序"""
    from federatedscope.contrib.utils.geometry_utils import eigendecompose

    # 创建测试协方差矩阵
    cov = torch.randn(10, 10)
    cov = cov @ cov.t()  # 确保正定

    eigenvalues, eigenvectors = eigendecompose(cov)

    # 检查是否升序
    is_ascending = all(eigenvalues[i] <= eigenvalues[i+1] for i in range(len(eigenvalues)-1))

    print("=" * 60)
    print("测试 1: 特征值排序")
    print("=" * 60)
    print(f"特征值: {eigenvalues[:5].tolist()}")
    print(f"是否升序: {is_ascending}")

    if is_ascending:
        print("✓ PASS: 特征值保持升序（正确）")
    else:
        print("✗ FAIL: 特征值不是升序（错误）")

    return is_ascending

def test_eigenvalue_scaling():
    """测试特征值缩放"""
    from federatedscope.contrib.utils.geometry_utils import _apply_eigenvalue_scaling

    # 创建升序特征值
    eigenvalues = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0,
                                1.5, 2.0, 2.5, 3.0, 3.5], dtype=torch.float32)

    scaled = _apply_eigenvalue_scaling(eigenvalues)

    # 前10个应该被缩放
    scale_factors_expected = torch.linspace(5.0, 1.0, 10)
    expected_first_10 = eigenvalues[:10] * scale_factors_expected

    match = torch.allclose(scaled[:10], expected_first_10, rtol=1e-4)

    print("\n" + "=" * 60)
    print("测试 2: 特征值缩放")
    print("=" * 60)
    print(f"原始前10个: {eigenvalues[:10].tolist()}")
    print(f"缩放后前10个: {scaled[:10].tolist()}")
    print(f"缩放因子: {scale_factors_expected.tolist()}")
    print(f"是否正确缩放最小的10个: {match}")

    if match:
        print("✓ PASS: 正确缩放最小的10个特征值")
    else:
        print("✗ FAIL: 缩放不正确")

    return match

def test_mlp_softmax():
    """测试 MLP 是否有 Softmax 层"""
    from federatedscope.contrib.trainer.ggeur_trainer import GGEURTrainer
    from federatedscope.core.configs.config import global_cfg
    import torch.nn as nn

    # 创建最小配置
    cfg = global_cfg.clone()
    cfg.ggeur.embedding_dim = 512
    cfg.ggeur.mlp_layers = 1
    cfg.model.num_classes = 10
    cfg.dataloader.batch_size = 16

    # 模拟创建 trainer
    class DummyContext:
        device = 'cpu'

    trainer = GGEURTrainer.__new__(GGEURTrainer)
    trainer._cfg = cfg
    trainer.ctx = DummyContext()
    trainer._build_mlp_classifier()

    # 检查最后一层是否是 Softmax
    layers = list(trainer.mlp_classifier.children())
    has_softmax = any(isinstance(layer, nn.Softmax) for layer in layers)
    last_is_softmax = isinstance(layers[-1], nn.Softmax) if layers else False

    print("\n" + "=" * 60)
    print("测试 3: MLP Softmax 层")
    print("=" * 60)
    print(f"MLP 结构: {trainer.mlp_classifier}")
    print(f"包含 Softmax: {has_softmax}")
    print(f"最后一层是 Softmax: {last_is_softmax}")

    if has_softmax and last_is_softmax:
        print("✓ PASS: MLP 正确添加了 Softmax 层")
    else:
        print("✗ FAIL: MLP 缺少 Softmax 层")

    return has_softmax and last_is_softmax

def test_config_files():
    """测试配置文件"""
    import yaml

    configs = {
        'Office-Home': 'scripts/example_configs/ggeur_officehome_lds.yaml',
        'Digits': 'scripts/example_configs/ggeur_digits_multi_domain.yaml',
        'PACS': 'scripts/example_configs/ggeur_pacs_multi_domain.yaml'
    }

    expected = {
        'Office-Home': {'local_update_steps': 1, 'optimizer_type': 'Adam', 'lr': 0.001},
        'Digits': {'local_update_steps': 10, 'optimizer_type': 'Adam', 'lr': 0.01},
        'PACS': {'local_update_steps': 10, 'optimizer_type': 'Adam', 'lr': 0.001}
    }

    print("\n" + "=" * 60)
    print("测试 4: 配置文件")
    print("=" * 60)

    all_correct = True

    for name, path in configs.items():
        with open(path, 'r') as f:
            config = yaml.safe_load(f)

        local_steps = config['train']['local_update_steps']
        optimizer_type = config['train']['optimizer']['type']
        lr = config['train']['optimizer']['lr']

        exp = expected[name]
        correct = (local_steps == exp['local_update_steps'] and
                  optimizer_type == exp['optimizer_type'] and
                  abs(lr - exp['lr']) < 1e-6)

        print(f"\n{name}:")
        print(f"  local_update_steps: {local_steps} (期望: {exp['local_update_steps']})")
        print(f"  optimizer.type: {optimizer_type} (期望: {exp['optimizer_type']})")
        print(f"  optimizer.lr: {lr} (期望: {exp['lr']})")

        if correct:
            print(f"  ✓ PASS")
        else:
            print(f"  ✗ FAIL")
            all_correct = False

    return all_correct

def main():
    print("GGEUR_Clip 修复验证")
    print("=" * 60)
    print()

    results = []

    try:
        results.append(("特征值排序", test_eigenvalue_order()))
    except Exception as e:
        print(f"✗ 测试1失败: {e}")
        results.append(("特征值排序", False))

    try:
        results.append(("特征值缩放", test_eigenvalue_scaling()))
    except Exception as e:
        print(f"✗ 测试2失败: {e}")
        results.append(("特征值缩放", False))

    try:
        results.append(("MLP Softmax", test_mlp_softmax()))
    except Exception as e:
        print(f"✗ 测试3失败: {e}")
        results.append(("MLP Softmax", False))

    try:
        results.append(("配置文件", test_config_files()))
    except Exception as e:
        print(f"✗ 测试4失败: {e}")
        results.append(("配置文件", False))

    # 总结
    print("\n" + "=" * 60)
    print("验证总结")
    print("=" * 60)

    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{name}: {status}")

    all_passed = all(r[1] for r in results)

    print("\n" + "=" * 60)
    if all_passed:
        print("所有测试通过！✓")
        print("可以开始运行实验了。")
    else:
        print("部分测试失败！✗")
        print("请检查上述错误。")
    print("=" * 60)

    return all_passed

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
