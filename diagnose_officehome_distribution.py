"""
诊断OfficeHome-LDS数据集分布
检查每个客户端的域分配和类别分布
"""

import sys
import os
import numpy as np
from collections import defaultdict, Counter

# Add FederatedScope to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from federatedscope.core.configs.config import CN, init_global_cfg
from federatedscope.core.auxiliaries.data_builder import get_data
from federatedscope.core.auxiliaries.utils import setup_seed


def analyze_officehome_distribution(config_path):
    """分析OfficeHome数据集在各个客户端的分布"""

    # 创建配置对象
    cfg = CN()

    # 初始化全局配置
    init_global_cfg(cfg)

    # 从文件加载配置
    cfg.merge_from_file(config_path)

    # 设置随机种子
    setup_seed(cfg.seed)

    print("="*80)
    print("OfficeHome-LDS 数据分布诊断")
    print("="*80)
    print(f"\n配置文件: {config_path}")
    print(f"数据集类型: {cfg.data.type}")
    print(f"客户端数量: {cfg.federate.client_num}")
    print(f"数据分割器: {cfg.data.splitter}")
    print(f"Dirichlet alpha: {getattr(cfg.data, 'dirichlet_alpha', 'NOT SET')}")

    # 加载数据
    print("\n加载数据...")
    data, modified_cfg = get_data(cfg)

    print(f"\n实际客户端数量: {len(data)}")
    print(f"客户端ID列表: {list(data.keys())}")

    # OfficeHome的4个域
    domains = ['Art', 'Clipart', 'Product', 'Real_World']

    # 分析每个客户端的数据
    print("\n" + "="*80)
    print("客户端数据分析")
    print("="*80)

    client_stats = {}

    for client_id in sorted(data.keys()):
        client_data = data[client_id]

        print(f"\n{'='*80}")
        print(f"客户端 {client_id}")
        print(f"{'='*80}")

        stats = {
            'train': {},
            'val': {},
            'test': {}
        }

        for split in ['train', 'val', 'test']:
            if split not in client_data:
                print(f"  {split}: 无数据")
                continue

            dataset = client_data[split]

            # 检查是否是DataLoader
            from torch.utils.data import DataLoader
            if isinstance(dataset, DataLoader):
                print(f"  {split}: 数据已被包装成DataLoader")
                # 从DataLoader中提取dataset
                dataset = dataset.dataset

            # 获取标签 - 支持多种数据集格式
            labels = None
            try:
                if hasattr(dataset, 'targets'):
                    labels = np.array(dataset.targets)
                elif hasattr(dataset, 'dataset') and hasattr(dataset.dataset, 'targets'):
                    # 处理Subset情况
                    if hasattr(dataset, 'indices'):
                        labels = np.array(dataset.dataset.targets)[dataset.indices]
                    else:
                        labels = np.array(dataset.dataset.targets)
                elif hasattr(dataset, '__getitem__') and hasattr(dataset, '__len__'):
                    # 尝试手动获取所有标签
                    print(f"  {split}: 手动提取标签（数据集大小: {len(dataset)}）...")
                    labels = []
                    sample_size = min(len(dataset), 2000)  # 最多检查2000个样本
                    for i in range(sample_size):
                        try:
                            _, label = dataset[i]
                            if isinstance(label, int) or isinstance(label, np.integer):
                                labels.append(int(label))
                            else:
                                labels.append(label)
                        except Exception as e:
                            print(f"  {split}: 获取样本 {i} 时出错: {e}")
                            if i > 10:  # 如果前10个都失败了，就放弃
                                break
                    if labels:
                        labels = np.array(labels)
                    else:
                        labels = None
            except Exception as e:
                print(f"  {split}: 获取标签时出错: {e}")
                import traceback
                traceback.print_exc()
                labels = None

            if labels is None or len(labels) == 0:
                print(f"  {split}: 无法获取标签")
                continue

            # 统计类别分布
            unique_labels, counts = np.unique(labels, return_counts=True)

            print(f"\n  {split.upper()} 集:")
            print(f"    样本总数: {len(labels)}")
            print(f"    类别数量: {len(unique_labels)} / 65")
            print(f"    类别ID: {unique_labels.tolist()[:10]}..." if len(unique_labels) > 10 else f"    类别ID: {unique_labels.tolist()}")

            # 计算每个类别的样本数统计
            print(f"    每类样本数统计:")
            print(f"      最小值: {counts.min()}")
            print(f"      最大值: {counts.max()}")
            print(f"      平均值: {counts.mean():.2f}")
            print(f"      中位数: {np.median(counts):.2f}")

            # 保存统计信息
            stats[split] = {
                'total_samples': len(labels),
                'num_classes': len(unique_labels),
                'class_labels': unique_labels.tolist(),
                'class_counts': dict(zip(unique_labels.tolist(), counts.tolist())),
                'min_samples_per_class': int(counts.min()),
                'max_samples_per_class': int(counts.max()),
                'mean_samples_per_class': float(counts.mean()),
            }

            # 显示详细的类别分布（前20个类别）
            if len(unique_labels) <= 20:
                print(f"    详细分布:")
                for label, count in zip(unique_labels, counts):
                    print(f"      类别 {label:2d}: {count:4d} 样本")

        client_stats[client_id] = stats

    # 总结
    print("\n" + "="*80)
    print("总结")
    print("="*80)

    print("\n各客户端训练集类别数量:")
    for client_id in sorted(client_stats.keys()):
        num_classes = client_stats[client_id]['train'].get('num_classes', 0)
        total_samples = client_stats[client_id]['train'].get('total_samples', 0)
        print(f"  客户端 {client_id}: {num_classes} 个类别, {total_samples} 个样本")

    # 检查是否符合LDS设置（每个客户端只有部分类别）
    print("\n关键发现:")
    all_clients_have_all_classes = all(
        client_stats[cid]['train'].get('num_classes', 0) == 65
        for cid in client_stats.keys()
    )

    if all_clients_have_all_classes:
        print("  ⚠️  所有客户端都有全部65个类别！")
        print("  ⚠️  这与OfficeHome-LDS的设置不符！")
        print("  ⚠️  LDS应该是每个客户端只有部分类别（Label Distribution Skew）")
    else:
        print("  ✓ 客户端类别分布不均（符合LDS设置）")

    # 检查客户端之间的类别重叠
    print("\n类别重叠分析:")
    train_classes_per_client = {
        cid: set(stats['train'].get('class_labels', []))
        for cid, stats in client_stats.items()
    }

    client_ids = sorted(train_classes_per_client.keys())
    for i, cid1 in enumerate(client_ids):
        for cid2 in client_ids[i+1:]:
            overlap = train_classes_per_client[cid1] & train_classes_per_client[cid2]
            total_union = train_classes_per_client[cid1] | train_classes_per_client[cid2]
            overlap_ratio = len(overlap) / len(total_union) * 100 if total_union else 0
            print(f"  客户端 {cid1} & 客户端 {cid2}: {len(overlap)} 个共同类别 ({overlap_ratio:.1f}% 重叠)")

    return client_stats


if __name__ == "__main__":
    config_path = "scripts/example_configs/ggeur_officehome_lds.yaml"

    if not os.path.exists(config_path):
        print(f"错误: 配置文件不存在: {config_path}")
        sys.exit(1)

    analyze_officehome_distribution(config_path)
