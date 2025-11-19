"""
依赖检查脚本 - 在运行训练前检查所有依赖是否已安装
"""
import sys

def check_dependencies():
    """检查所有必需的依赖"""
    print("=" * 80)
    print("FederatedScope 依赖检查")
    print("=" * 80)

    dependencies = {
        'numpy': 'NumPy',
        'scipy': 'SciPy',
        'torch': 'PyTorch',
        'torchvision': 'TorchVision',
        'yaml': 'PyYAML',
        'PIL': 'Pillow',
        'pandas': 'Pandas (可选)',
        'matplotlib': 'Matplotlib (可选)',
    }

    missing = []
    installed = []

    for module, name in dependencies.items():
        try:
            __import__(module)
            version = None
            try:
                mod = sys.modules[module]
                if hasattr(mod, '__version__'):
                    version = mod.__version__
            except:
                pass

            version_str = f" (v{version})" if version else ""
            installed.append(f"✓ {name}{version_str}")
        except ImportError:
            missing.append(f"✗ {name} - 未安装")

    print("\n已安装的依赖:")
    for item in installed:
        print(f"  {item}")

    if missing:
        print("\n缺失的依赖:")
        for item in missing:
            print(f"  {item}")
        print("\n请运行以下命令安装缺失的依赖:")
        print("  pip install scipy numpy torch torchvision pillow pyyaml")
        return False
    else:
        print("\n✓ 所有必需依赖已安装!")
        return True


def check_cuda():
    """检查CUDA是否可用"""
    print("\n" + "=" * 80)
    print("CUDA检查")
    print("=" * 80)

    try:
        import torch
        if torch.cuda.is_available():
            print(f"✓ CUDA可用")
            print(f"  CUDA版本: {torch.version.cuda}")
            print(f"  可用GPU数量: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
            return True
        else:
            print("✗ CUDA不可用 (将使用CPU)")
            print("  如需使用GPU，请安装支持CUDA的PyTorch版本")
            return False
    except ImportError:
        print("✗ PyTorch未安装")
        return False


def check_config_registration():
    """检查配置是否正确注册"""
    print("\n" + "=" * 80)
    print("配置注册检查")
    print("=" * 80)

    try:
        from federatedscope.core.configs.config import global_cfg
        cfg = global_cfg.clone()

        required_configs = ['ondemfl', 'fedlsa', 'cross_domain_adaptive']

        for config_name in required_configs:
            if hasattr(cfg, config_name):
                print(f"✓ {config_name} 配置已注册")
            else:
                print(f"✗ {config_name} 配置未注册")
                return False

        print("\n✓ 所有配置项已正确注册!")
        return True

    except Exception as e:
        print(f"✗ 配置检查失败: {e}")
        print("\n可能的原因:")
        print("  1. 缺少依赖包 (特别是scipy)")
        print("  2. 配置文件有语法错误")
        return False


def check_dataset():
    """检查数据集是否存在"""
    print("\n" + "=" * 80)
    print("数据集检查")
    print("=" * 80)

    import os

    # Try both common paths
    dataset_paths = ["root/data/office_caltech_10", "data/office_caltech_10"]
    dataset_path = None
    for path in dataset_paths:
        if os.path.exists(path):
            dataset_path = path
            break

    if dataset_path is None:
        dataset_path = "root/data/office_caltech_10"  # Use config default
    domains = ['amazon', 'webcam', 'dslr', 'caltech']

    if not os.path.exists(dataset_path):
        print(f"✗ 数据集目录不存在: {dataset_path}")
        print("\n请下载Office-Caltech-10数据集:")
        print("  python scripts/download_office_caltech.py")
        return False

    missing_domains = []
    for domain in domains:
        domain_path = os.path.join(dataset_path, domain)
        if os.path.exists(domain_path):
            print(f"✓ {domain} 域存在")
        else:
            missing_domains.append(domain)
            print(f"✗ {domain} 域不存在")

    if missing_domains:
        print(f"\n✗ 缺少域: {', '.join(missing_domains)}")
        return False
    else:
        print("\n✓ 所有域的数据都存在!")
        return True


def check_config_file():
    """检查配置文件是否存在"""
    print("\n" + "=" * 80)
    print("配置文件检查")
    print("=" * 80)

    import os

    config_path = "scripts/example_configs/cross_domain_adaptive_office_caltech.yaml"

    if os.path.exists(config_path):
        print(f"✓ 配置文件存在: {config_path}")

        # 检查关键配置项
        try:
            import yaml
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)

            # 检查make_global_eval
            if config.get('federate', {}).get('make_global_eval') == True:
                print("✓ make_global_eval 已启用 (服务器端评估)")
            else:
                print("✗ make_global_eval 未启用")
                print("  建议设置: federate.make_global_eval = True")

            # 检查数据集类型
            if config.get('data', {}).get('type') == 'office_caltech':
                print("✓ 数据集类型正确: office_caltech")
            else:
                print("✗ 数据集类型不正确")

            return True
        except Exception as e:
            print(f"✗ 配置文件解析失败: {e}")
            return False
    else:
        print(f"✗ 配置文件不存在: {config_path}")
        return False


def main():
    """主函数"""
    print("\n开始环境检查...\n")

    results = {
        "依赖": check_dependencies(),
        "CUDA": check_cuda(),
        "配置注册": check_config_registration(),
        "数据集": check_dataset(),
        "配置文件": check_config_file(),
    }

    print("\n" + "=" * 80)
    print("检查总结")
    print("=" * 80)

    for name, status in results.items():
        status_str = "✓ 通过" if status else "✗ 失败"
        print(f"{name}: {status_str}")

    all_passed = all(results.values())

    if all_passed:
        print("\n" + "=" * 80)
        print("✓ 所有检查通过! 可以开始训练")
        print("=" * 80)
        print("\n运行训练命令:")
        print("  python federatedscope/main.py --cfg scripts/example_configs/cross_domain_adaptive_office_caltech.yaml")
        print("\n或者:")
        print("  python run_cross_domain_adaptive.py --cfg scripts/example_configs/cross_domain_adaptive_office_caltech.yaml")
        return 0
    else:
        print("\n" + "=" * 80)
        print("✗ 部分检查失败，请先解决上述问题")
        print("=" * 80)
        print("\n详细解决方案请参考: 运行指南.md")
        return 1


if __name__ == "__main__":
    sys.exit(main())
