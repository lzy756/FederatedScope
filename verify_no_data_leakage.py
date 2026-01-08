"""
验证服务器测试集和客户端数据之间没有数据泄露

这个脚本会：
1. 加载服务器的held-out测试集
2. 加载客户端的训练/验证/测试数据
3. 检查是否有重叠的样本
"""

import sys
import os

# Add FederatedScope to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from federatedscope.cv.dataset.office_caltech import (
    load_balanced_office_caltech_data,
    load_office_caltech_domain_data
)


def verify_no_overlap_office_caltech():
    """验证 Office-Caltech 数据集没有泄露"""
    print("="*80)
    print("验证 Office-Caltech 数据集 - 服务器测试集与客户端数据无重叠")
    print("="*80)

    root = '/root/data/office_caltech_10'
    seed = 123
    samples_per_class = 20
    splits = (0.8, 0.1, 0.1)

    # 1. 加载服务器held-out测试集
    print("\n[1] 加载服务器held-out测试集...")
    result = load_balanced_office_caltech_data(
        root=root,
        samples_per_class=samples_per_class,
        transform=None,
        seed=seed
    )

    server_datasets = result['datasets']
    excluded_indices = result['excluded_indices']

    print(f"服务器测试集包含 {len(server_datasets)} 个域")

    # 统计服务器测试集样本路径
    server_test_paths = {}
    for domain, dataset in server_datasets.items():
        paths = set(dataset.data)
        server_test_paths[domain] = paths
        print(f"  域 '{domain}': {len(paths)} 个样本, {len(excluded_indices[domain])} 个排除索引")

    # 2. 加载客户端数据（使用exclude_indices）
    print("\n[2] 加载客户端数据（使用exclude_indices）...")
    client_data_paths = {}

    for domain in ['amazon', 'webcam', 'dslr', 'caltech']:
        print(f"\n  加载域 '{domain}'...")
        exclude_indices_for_domain = excluded_indices.get(domain, None)

        domain_data = load_office_caltech_domain_data(
            root=root,
            domain=domain,
            splits=splits,
            transform=None,
            seed=seed,
            exclude_indices=exclude_indices_for_domain
        )

        # 收集所有客户端数据路径
        all_client_paths = set()
        for split_name in ['train', 'val', 'test']:
            split_paths = set(domain_data[split_name].data)
            all_client_paths.update(split_paths)
            print(f"    {split_name}: {len(split_paths)} 个样本")

        client_data_paths[domain] = all_client_paths
        print(f"    总计: {len(all_client_paths)} 个客户端样本")

    # 3. 检查重叠
    print("\n[3] 检查服务器测试集和客户端数据是否有重叠...")
    has_leakage = False

    for domain in server_test_paths.keys():
        server_paths = server_test_paths[domain]
        client_paths = client_data_paths[domain]

        overlap = server_paths.intersection(client_paths)

        if overlap:
            has_leakage = True
            print(f"\n  ❌ 域 '{domain}': 发现 {len(overlap)} 个重叠样本!")
            print(f"     示例: {list(overlap)[:3]}")
        else:
            print(f"  ✓ 域 '{domain}': 无重叠 (服务器: {len(server_paths)}, 客户端: {len(client_paths)})")

    # 4. 总结
    print("\n" + "="*80)
    if has_leakage:
        print("❌ 验证失败: 发现数据泄露!")
        return False
    else:
        print("✓ 验证成功: 服务器测试集和客户端数据完全分离，无数据泄露!")
        return True


def verify_no_overlap_without_exclusion():
    """验证不使用exclude_indices时确实存在重叠（对照组）"""
    print("\n\n" + "="*80)
    print("对照实验: 不使用exclude_indices时的数据重叠情况")
    print("="*80)

    root = '/root/data/office_caltech_10'
    seed = 123
    samples_per_class = 20
    splits = (0.8, 0.1, 0.1)

    # 1. 加载服务器测试集
    print("\n[1] 加载服务器held-out测试集...")
    result = load_balanced_office_caltech_data(
        root=root,
        samples_per_class=samples_per_class,
        transform=None,
        seed=seed
    )

    server_datasets = result['datasets']
    server_test_paths = {}
    for domain, dataset in server_datasets.items():
        server_test_paths[domain] = set(dataset.data)

    # 2. 加载客户端数据（不使用exclude_indices）
    print("\n[2] 加载客户端数据（不使用exclude_indices - 旧方式）...")
    client_data_paths = {}

    for domain in ['amazon', 'webcam', 'dslr', 'caltech']:
        domain_data = load_office_caltech_domain_data(
            root=root,
            domain=domain,
            splits=splits,
            transform=None,
            seed=seed,
            exclude_indices=None  # 不排除
        )

        all_client_paths = set()
        for split_name in ['train', 'val', 'test']:
            all_client_paths.update(domain_data[split_name].data)

        client_data_paths[domain] = all_client_paths

    # 3. 检查重叠
    print("\n[3] 检查重叠...")
    total_overlap = 0

    for domain in server_test_paths.keys():
        overlap = server_test_paths[domain].intersection(client_data_paths[domain])
        total_overlap += len(overlap)

        if overlap:
            print(f"  域 '{domain}': {len(overlap)} 个重叠样本")

    print("\n" + "="*80)
    if total_overlap > 0:
        print(f"✓ 对照实验成功: 不使用exclude_indices时发现 {total_overlap} 个重叠样本")
        print("  这证明了exclude_indices机制的必要性!")
    else:
        print("⚠ 意外: 即使不使用exclude_indices也没有重叠，需要检查逻辑")

    return total_overlap > 0


if __name__ == '__main__':
    print("Office-Caltech 数据泄露验证工具\n")

    # 主验证: 使用exclude_indices应该无重叠
    success = verify_no_overlap_office_caltech()

    # 对照实验: 不使用exclude_indices应该有重叠
    control_success = verify_no_overlap_without_exclusion()

    print("\n\n" + "="*80)
    print("最终结果:")
    print("="*80)

    if success and control_success:
        print("✓✓✓ 完美! exclude_indices机制工作正常!")
        print("  - 使用exclude_indices: 无数据泄露")
        print("  - 不使用exclude_indices: 有数据泄露（预期行为）")
        sys.exit(0)
    elif success and not control_success:
        print("⚠ 使用exclude_indices无重叠，但对照组也无重叠，需要检查")
        sys.exit(1)
    else:
        print("❌ 验证失败，仍然存在数据泄露!")
        sys.exit(1)
