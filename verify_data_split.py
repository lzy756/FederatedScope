"""
Verify actual data split during GGEUR_Clip training
This script simulates the exact data loading process used in training
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def verify_ggeur_data_loading():
    """Verify the actual data loading process"""
    print("=" * 80)
    print("Verifying GGEUR_Clip Data Loading Process")
    print("=" * 80)

    # Simulate config values from ggeur_officehome.yaml
    class MockConfig:
        class Data:
            type = 'office-home'
            root = 'OfficeHomeDataset_10072016'
            splits = [0.7, 0.0, 0.3]
        class Federate:
            client_num = 12
        data = Data()
        federate = Federate()
        seed = 42  # From config file

    config = MockConfig()

    print(f"\nConfig values:")
    print(f"  data.type = {config.data.type}")
    print(f"  data.splits = {config.data.splits}")
    print(f"  seed = {config.seed}")
    print(f"  client_num = {config.federate.client_num}")

    # Import the data loading function
    from federatedscope.cv.dataset.office_home import OfficeHome, load_office_home_domain_data

    # Simulate ggeur_data.py loading
    print("\n" + "-" * 40)
    print("Simulating ggeur_data.py (client data loading)")
    print("-" * 40)

    splits = tuple(config.data.splits)
    seed = config.seed

    print(f"  Using splits={splits}, seed={seed}")

    all_train_paths = set()
    all_test_paths = set()

    for domain in OfficeHome.DOMAINS:
        domain_data = load_office_home_domain_data(
            root=config.data.root,
            domain=domain,
            splits=splits,
            transform=None,
            seed=seed
        )

        train_paths = set(domain_data['train'].data)
        test_paths = set(domain_data['test'].data)

        all_train_paths.update(train_paths)
        all_test_paths.update(test_paths)

        print(f"  {domain}: train={len(train_paths)}, test={len(test_paths)}")

    # Simulate ggeur_server.py loading
    print("\n" + "-" * 40)
    print("Simulating ggeur_server.py (server test data loading)")
    print("-" * 40)

    # This is what server does in _load_test_data_and_features
    data_type = config.data.type.lower()
    if hasattr(config.data, 'splits'):
        server_splits = tuple(config.data.splits)
    else:
        if 'office' in data_type and 'home' in data_type:
            server_splits = (0.7, 0.0, 0.3)
        else:
            server_splits = (0.8, 0.1, 0.1)

    server_seed = config.seed if hasattr(config, 'seed') else 123

    print(f"  Using splits={server_splits}, seed={server_seed}")

    server_test_paths = set()

    for domain in OfficeHome.DOMAINS:
        test_dataset = OfficeHome(
            root=config.data.root,
            domain=domain,
            split='test',
            transform=None,
            train_ratio=server_splits[0],
            val_ratio=server_splits[1],
            seed=server_seed
        )

        server_test_paths.update(test_dataset.data)
        print(f"  {domain}: test={len(test_dataset)}")

    # Check overlap
    print("\n" + "-" * 40)
    print("Checking for Data Leakage")
    print("-" * 40)

    train_vs_server_test = all_train_paths & server_test_paths
    client_test_vs_server_test = all_test_paths & server_test_paths

    print(f"  Client train paths: {len(all_train_paths)}")
    print(f"  Client test paths: {len(all_test_paths)}")
    print(f"  Server test paths: {len(server_test_paths)}")
    print(f"  Overlap (train vs server test): {len(train_vs_server_test)}")
    print(f"  Overlap (client test vs server test): {len(client_test_vs_server_test)}")

    if train_vs_server_test:
        overlap_ratio = len(train_vs_server_test) / len(server_test_paths) * 100
        print(f"\n  !!! DATA LEAKAGE DETECTED !!!")
        print(f"  {overlap_ratio:.1f}% of server test data is in client training set!")
    elif client_test_vs_server_test == server_test_paths:
        print(f"\n  [OK] Server test set is identical to client test set")
        print(f"  [OK] No training data leakage")
    else:
        print(f"\n  WARNING: Server test set differs from client test set")
        print(f"  This might indicate a configuration mismatch")


def check_hasattr_behavior():
    """Check hasattr behavior with FederatedScope config"""
    print("\n" + "=" * 80)
    print("Checking hasattr Behavior")
    print("=" * 80)

    try:
        from federatedscope.core.configs.config import CN, global_cfg
        print(f"  global_cfg type: {type(global_cfg)}")
        print(f"  hasattr(global_cfg, 'seed'): {hasattr(global_cfg, 'seed')}")
        if hasattr(global_cfg, 'seed'):
            print(f"  global_cfg.seed: {global_cfg.seed}")
    except Exception as e:
        print(f"  Error: {e}")


if __name__ == '__main__':
    verify_ggeur_data_loading()
    check_hasattr_behavior()
