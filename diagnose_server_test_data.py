"""
诊断服务器端测试数据加载问题
"""
import os
import sys
import logging

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_data_path(data_root):
    """检查数据路径是否存在"""
    print("\n" + "="*70)
    print("检查数据路径")
    print("="*70)

    if not os.path.exists(data_root):
        print(f"[错误] 数据根目录不存在: {data_root}")
        return False

    print(f"[OK] 数据根目录存在: {data_root}")

    # 检查各个域的目录
    domains = ['amazon', 'webcam', 'dslr', 'caltech']
    all_exist = True

    for domain in domains:
        domain_path = os.path.join(data_root, domain)
        if os.path.exists(domain_path):
            # 统计该域的图片数量
            total_images = 0
            for class_name in os.listdir(domain_path):
                class_path = os.path.join(domain_path, class_name)
                if os.path.isdir(class_path):
                    images = [f for f in os.listdir(class_path)
                             if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                    total_images += len(images)
            print(f"  [OK] {domain:10s}: {total_images:4d} images")
        else:
            print(f"  [错误] {domain:10s}: 目录不存在")
            all_exist = False

    return all_exist


def test_balanced_data_loading(data_root, samples_per_class=10):
    """测试均衡数据加载"""
    print("\n" + "="*70)
    print(f"测试均衡数据加载 (每类 {samples_per_class} 个样本)")
    print("="*70)

    try:
        from federatedscope.cv.dataset.office_caltech import load_balanced_office_caltech_data

        datasets = load_balanced_office_caltech_data(
            root=data_root,
            samples_per_class=samples_per_class,
            seed=123
        )

        print(f"\n[OK] 成功加载 {len(datasets)} 个域的均衡测试集")

        for domain, dataset in datasets.items():
            print(f"\n域: {domain}")
            print(f"  - 样本总数: {len(dataset)}")
            print(f"  - 类别标签: {set(dataset.targets)}")

            # 检查类别分布
            from collections import Counter
            label_counts = Counter(dataset.targets)
            print(f"  - 类别分布: {dict(label_counts)}")

            # 验证是否均衡
            counts = list(label_counts.values())
            if len(set(counts)) == 1 and counts[0] == samples_per_class:
                print(f"  [OK] 类别完全均衡 (每类 {samples_per_class} 个样本)")
            else:
                print(f"  [警告] 类别不均衡！期望每类 {samples_per_class} 个，实际: {counts}")

        return True, datasets

    except Exception as e:
        print(f"\n[错误] 加载失败: {e}")
        import traceback
        traceback.print_exc()
        return False, None


def test_data_loader(datasets):
    """测试 DataLoader"""
    print("\n" + "="*70)
    print("测试 DataLoader")
    print("="*70)

    try:
        from torch.utils.data import DataLoader

        for domain, dataset in datasets.items():
            loader = DataLoader(dataset, batch_size=4, shuffle=False)

            # 获取第一个batch
            batch = next(iter(loader))
            images, labels = batch

            print(f"\n域: {domain}")
            print(f"  - Batch 图像形状: {images.shape}")
            print(f"  - Batch 标签形状: {labels.shape}")
            print(f"  - Batch 标签值: {labels.tolist()}")
            print(f"  [OK] DataLoader 正常工作")

        return True

    except Exception as e:
        print(f"\n[错误] DataLoader 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    # 从配置文件读取数据路径
    import yaml

    config_path = 'scripts/example_configs/cross_domain_adaptive_office_caltech.yaml'

    print("="*70)
    print("Office-Caltech 服务器端测试数据诊断")
    print("="*70)

    # 读取配置
    if os.path.exists(config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        data_root = config['data']['root']
        samples_per_class = config['data'].get('server_test_samples_per_class', 10)
        print(f"\n从配置文件读取:")
        print(f"  - 数据路径: {data_root}")
        print(f"  - 每类样本数: {samples_per_class}")
    else:
        print(f"\n[警告] 配置文件不存在: {config_path}")
        data_root = '/root/data/office_caltech_10'
        samples_per_class = 10
        print(f"使用默认值:")
        print(f"  - 数据路径: {data_root}")
        print(f"  - 每类样本数: {samples_per_class}")

    # 步骤1: 检查数据路径
    path_ok = check_data_path(data_root)

    if not path_ok:
        print("\n" + "="*70)
        print("[错误] 数据路径检查失败！")
        print("请确保:")
        print(f"  1. 数据已下载到: {data_root}")
        print("  2. 目录结构正确:")
        print("     {data_root}/")
        print("       ├── amazon/")
        print("       ├── webcam/")
        print("       ├── dslr/")
        print("       └── caltech/")
        print("="*70)
        sys.exit(1)

    # 步骤2: 测试均衡数据加载
    load_ok, datasets = test_balanced_data_loading(data_root, samples_per_class)

    if not load_ok:
        print("\n" + "="*70)
        print("[错误] 均衡数据加载失败！")
        print("="*70)
        sys.exit(1)

    # 步骤3: 测试 DataLoader
    loader_ok = test_data_loader(datasets)

    # 总结
    print("\n" + "="*70)
    print("诊断总结")
    print("="*70)
    print(f"  [{'OK' if path_ok else 'FAIL'}] 数据路径检查")
    print(f"  [{'OK' if load_ok else 'FAIL'}] 均衡数据加载")
    print(f"  [{'OK' if loader_ok else 'FAIL'}] DataLoader 测试")

    if path_ok and load_ok and loader_ok:
        print("\n[成功] 所有检查通过！服务器端测试数据加载应该可以正常工作。")
        print("\n下一步:")
        print("  1. 确保配置文件中 federate.make_global_eval = True")
        print("  2. 确保配置文件中 data.server_test_samples_per_class = 10")
        print("  3. 运行训练查看服务器端评估")
    else:
        print("\n[失败] 某些检查未通过，请先解决数据问题。")

    print("="*70)
