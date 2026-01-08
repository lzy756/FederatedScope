"""
验证LDS（Label Distribution Skew）实现是否正确
检查每个客户端的类别分布是否高度不均衡
"""

import sys
import os
import numpy as np
from collections import Counter

# Add FederatedScope to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from federatedscope.core.configs.config import CN, init_global_cfg
from federatedscope.core.auxiliaries.data_builder import get_data
from federatedscope.core.auxiliaries.utils import setup_seed


def verify_lds_implementation(config_path):
    """验证LDS实现是否正确"""

    # 创建配置对象
    cfg = CN()
    init_global_cfg(cfg)
    cfg.merge_from_file(config_path)
    setup_seed(cfg.seed)

    print("="*80)
    print("验证 OfficeHome-LDS 实现")
    print("="*80)
    print(f"\n配置文件: {config_path}")
    print(f"数据集类型: {cfg.data.type}")
    print(f"客户端数量: {cfg.federate.client_num}")
    print(f"数据分割器: {cfg.data.splitter}")

    if cfg.data.splitter == 'lda_domain':
        alpha = cfg.data.splitter_args[0].get('alpha', 0.1) if cfg.data.splitter_args else 0.1
        print(f"LDS Alpha: {alpha}")
        print("\n✓ LDS模式已启用")
    else:
        print(f"\n⚠️  警告: 未使用LDS模式！当前splitter={cfg.data.splitter}")
        return

    # 加载数据
    print("\n加载数据...")
    data, modified_cfg = get_data(cfg)

    print(f"\n实际客户端数量: {len(data)}")

    # 分析每个客户端的训练集
    print("\n" + "="*80)
    print("客户端训练集分析（LDS验证）")
    print("="*80)

    all_client_stats = {}

    for client_id in sorted(data.keys()):
        client_data = data[client_id]

        if 'train' not in client_data:
            print(f"\n客户端 {client_id}: 无训练数据")
            continue

        dataset = client_data['train']

        # 从DataLoader中提取dataset
        from torch.utils.data import DataLoader
        if isinstance(dataset, DataLoader):
            dataset = dataset.dataset

        # 提取标签
        print(f"\n{'='*80}")
        print(f"客户端 {client_id}")
        print(f"{'='*80}")

        try:
            # 尝试直接获取标签
            if hasattr(dataset, 'targets'):
                labels = np.array(dataset.targets)
            elif hasattr(dataset, 'dataset') and hasattr(dataset.dataset, 'targets'):
                if hasattr(dataset, 'indices'):
                    labels = np.array(dataset.dataset.targets)[dataset.indices]
                else:
                    labels = np.array(dataset.dataset.targets)
            else:
                # 手动提取
                print(f"手动提取标签...")
                labels = []
                for i in range(len(dataset)):
                    _, label = dataset[i]
                    labels.append(int(label))
                labels = np.array(labels)

            # 统计每个类的样本数
            unique_labels, counts = np.unique(labels, return_counts=True)
            class_counts = dict(zip(unique_labels.tolist(), counts.tolist()))

            # 填充缺失的类（样本数为0）
            for class_id in range(65):
                if class_id not in class_counts:
                    class_counts[class_id] = 0

            # 排序
            class_counts = dict(sorted(class_counts.items()))
            counts_array = np.array(list(class_counts.values()))

            print(f"总样本数: {len(labels)}")
            print(f"非零类别数: {len(unique_labels)} / 65")
            print(f"零样本类别数: {sum(1 for c in class_counts.values() if c == 0)}")
            print(f"\n每类样本数统计:")
            print(f"  最小值: {counts_array.min()}")
            print(f"  最大值: {counts_array.max()}")
            print(f"  平均值: {counts_array.mean():.2f}")
            print(f"  中位数: {np.median(counts_array):.2f}")
            print(f"  标准差: {counts_array.std():.2f}")

            # 显示样本数前10和后10的类
            sorted_classes = sorted(class_counts.items(), key=lambda x: x[1], reverse=True)

            print(f"\n样本数最多的10个类:")
            for class_id, count in sorted_classes[:10]:
                print(f"  类别 {class_id:2d}: {count:4d} 样本")

            print(f"\n样本数最少的10个类:")
            for class_id, count in sorted_classes[-10:]:
                print(f"  类别 {class_id:2d}: {count:4d} 样本")

            # 保存统计
            all_client_stats[client_id] = {
                'total_samples': len(labels),
                'num_nonzero_classes': len(unique_labels),
                'num_zero_classes': sum(1 for c in class_counts.values() if c == 0),
                'min_samples': counts_array.min(),
                'max_samples': counts_array.max(),
                'mean_samples': counts_array.mean(),
                'std_samples': counts_array.std(),
                'class_counts': class_counts
            }

        except Exception as e:
            print(f"错误: {e}")
            import traceback
            traceback.print_exc()

    # 总结
    print("\n" + "="*80)
    print("LDS验证结果")
    print("="*80)

    if len(all_client_stats) == 0:
        print("⚠️  无法获取客户端统计信息")
        return

    print("\n各客户端概览:")
    print(f"{'客户端':<10} {'总样本':<10} {'非零类':<10} {'零类':<10} {'最小':<10} {'最大':<10} {'标准差':<10}")
    print("-" * 80)

    for client_id in sorted(all_client_stats.keys()):
        stats = all_client_stats[client_id]
        print(f"{client_id:<10} {stats['total_samples']:<10} {stats['num_nonzero_classes']:<10} "
              f"{stats['num_zero_classes']:<10} {stats['min_samples']:<10} {stats['max_samples']:<10} "
              f"{stats['std_samples']:<10.2f}")

    # 验证LDS特性
    print("\nLDS特性验证:")

    # 1. 检查是否有高度不均衡的类别分布
    high_std_count = sum(1 for stats in all_client_stats.values() if stats['std_samples'] > 10)
    print(f"1. 高标准差客户端数（>10）: {high_std_count} / {len(all_client_stats)}")

    # 2. 检查是否有零样本类别
    has_zero_classes = sum(1 for stats in all_client_stats.values() if stats['num_zero_classes'] > 0)
    print(f"2. 有零样本类别的客户端: {has_zero_classes} / {len(all_client_stats)}")

    # 3. 检查样本数范围
    max_range = max(stats['max_samples'] - stats['min_samples'] for stats in all_client_stats.values())
    print(f"3. 最大类别样本数差异: {max_range}")

    # 判断LDS是否生效
    print("\n" + "="*80)
    if high_std_count >= len(all_client_stats) * 0.75 and has_zero_classes > 0:
        print("✓ LDS实现正确！数据分布高度不均衡")
        print("  - 大部分客户端有高标准差的类别分布")
        print("  - 存在零样本类别")
        print("  - 符合论文的LDS设置")
    else:
        print("⚠️  LDS可能未正确实现")
        print(f"  - 高标准差客户端数: {high_std_count} / {len(all_client_stats)}")
        print(f"  - 有零样本类别的客户端: {has_zero_classes} / {len(all_client_stats)}")
    print("="*80)


if __name__ == "__main__":
    config_path = "scripts/example_configs/ggeur_officehome_lds.yaml"

    if not os.path.exists(config_path):
        print(f"错误: 配置文件不存在: {config_path}")
        sys.exit(1)

    verify_lds_implementation(config_path)
