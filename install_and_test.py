#!/usr/bin/env python
"""安装和测试FederatedScope环境"""
import subprocess
import sys

def install_package(package):
    """安装单个包"""
    print(f"Installing {package}...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package, "-q"])
        print(f"[OK] {package} installed")
        return True
    except subprocess.CalledProcessError:
        print(f"[ERROR] Failed to install {package}")
        return False

def test_import(module_name):
    """测试导入"""
    try:
        __import__(module_name)
        print(f"[OK] {module_name} can be imported")
        return True
    except ImportError as e:
        print(f"[ERROR] Cannot import {module_name}: {e}")
        return False

def main():
    print("="*80)
    print("Installing FederatedScope dependencies...")
    print("="*80 + "\n")

    # 基础依赖
    packages = [
        "numpy<1.23.0",
        "scikit-learn==1.0.2",
        "scipy==1.7.3",
        "pandas",
        "pyyaml>=5.1",
        "protobuf==3.19.4",
        "grpcio>=1.45.0",
        "grpcio-tools",
    ]

    for pkg in packages:
        install_package(pkg)

    print("\n" + "="*80)
    print("Testing imports...")
    print("="*80 + "\n")

    # 测试导入
    test_modules = [
        "numpy",
        "scipy",
        "sklearn",
        "pandas",
        "yaml",
        "google.protobuf",
        "grpc",
    ]

    all_ok = True
    for mod in test_modules:
        if not test_import(mod):
            all_ok = False

    if all_ok:
        print("\n[OK] All dependencies installed successfully!")

        # 测试federatedscope配置
        print("\nTesting FederatedScope configuration...")
        sys.path.insert(0, '.')
        try:
            from federatedscope.core.configs.config import global_cfg
            print("[OK] FederatedScope config loaded")
            print(f"[OK] Has fedlsa: {hasattr(global_cfg, 'fedlsa')}")
            print(f"[OK] Has ondemfl: {hasattr(global_cfg, 'ondemfl')}")
            print(f"[OK] Has cross_domain_adaptive: {hasattr(global_cfg, 'cross_domain_adaptive')}")
        except Exception as e:
            print(f"[ERROR] Failed to load FederatedScope config: {e}")
            return False
    else:
        print("\n[ERROR] Some dependencies failed to install")
        return False

    return True

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
