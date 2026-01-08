"""
Diagnostic script: Check GGEUR_Clip train/test data overlap
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def check_office_home_overlap():
    """Check Office-Home dataset for train/test overlap"""
    from federatedscope.cv.dataset.office_home import OfficeHome

    print("=" * 80)
    print("Checking Office-Home Dataset")
    print("=" * 80)

    root = 'OfficeHomeDataset_10072016'
    splits = (0.7, 0.0, 0.3)
    seed = 42

    if not os.path.exists(root):
        print(f"Dataset directory not found: {root}")
        return 0

    domains = OfficeHome.DOMAINS
    total_overlap = 0
    total_train = 0
    total_test = 0

    for domain in domains:
        print(f"\n--- Checking domain: {domain} ---")

        train_dataset = OfficeHome(
            root=root, domain=domain, split='train',
            transform=None, train_ratio=splits[0], val_ratio=splits[1], seed=seed
        )

        test_dataset = OfficeHome(
            root=root, domain=domain, split='test',
            transform=None, train_ratio=splits[0], val_ratio=splits[1], seed=seed
        )

        train_paths = set(train_dataset.data)
        test_paths = set(test_dataset.data)
        overlap = train_paths & test_paths

        print(f"  Train size: {len(train_paths)}")
        print(f"  Test size: {len(test_paths)}")
        print(f"  Overlap count: {len(overlap)}")

        if overlap:
            print(f"  !!! DATA LEAKAGE !!!")
            for path in list(overlap)[:3]:
                print(f"    - {os.path.basename(path)}")
        else:
            print(f"  [OK] No overlap")

        total_overlap += len(overlap)
        total_train += len(train_paths)
        total_test += len(test_paths)

    print("\n" + "=" * 80)
    print(f"Office-Home Summary:")
    print(f"  Total train: {total_train}, Total test: {total_test}")
    print(f"  Total overlap: {total_overlap}")
    print("=" * 80)

    return total_overlap


def check_different_seeds():
    """Check if different seeds cause overlap"""
    from federatedscope.cv.dataset.office_home import OfficeHome

    print("\n" + "=" * 80)
    print("Checking Different Seeds Impact")
    print("=" * 80)

    root = 'OfficeHomeDataset_10072016'
    splits = (0.7, 0.0, 0.3)
    domain = 'Art'

    if not os.path.exists(root):
        print(f"Dataset directory not found: {root}")
        return

    # Client uses seed=42, server might use seed=123 (old default)
    client_seed = 42
    server_seed = 123

    print(f"\nClient seed={client_seed}, Server seed={server_seed}")

    train_client = OfficeHome(
        root=root, domain=domain, split='train',
        transform=None, train_ratio=splits[0], val_ratio=splits[1], seed=client_seed
    )

    test_server = OfficeHome(
        root=root, domain=domain, split='test',
        transform=None, train_ratio=splits[0], val_ratio=splits[1], seed=server_seed
    )

    train_paths = set(train_client.data)
    test_paths = set(test_server.data)
    overlap = train_paths & test_paths

    print(f"  Client train (seed={client_seed}): {len(train_paths)} samples")
    print(f"  Server test (seed={server_seed}): {len(test_paths)} samples")
    print(f"  Overlap: {len(overlap)}")

    if overlap:
        overlap_ratio = len(overlap) / len(test_paths) * 100
        print(f"\n  !!! DATA LEAKAGE FOUND !!!")
        print(f"  {overlap_ratio:.1f}% of test data appears in train set!")
        return len(overlap)
    else:
        print(f"  [OK] No overlap")
        return 0


def check_actual_server_loading():
    """Check how server actually loads test data"""
    print("\n" + "=" * 80)
    print("Checking Server Loading Logic")
    print("=" * 80)

    # Read ggeur_server.py to understand the actual loading
    server_file = 'federatedscope/contrib/worker/ggeur_server.py'
    if os.path.exists(server_file):
        with open(server_file, 'r', encoding='utf-8') as f:
            content = f.read()

        # Check for seed usage
        if 'seed = self._cfg.seed' in content:
            print("  Server uses: seed = self._cfg.seed (with fallback)")
        if 'seed=123' in content or 'seed = 123' in content:
            print("  WARNING: Server has default seed=123")

        # Check for splits
        if "splits = (0.7, 0.0, 0.3)" in content:
            print("  Server default splits: (0.7, 0.0, 0.3)")
        if "splits = (0.8, 0.1, 0.1)" in content:
            print("  WARNING: Server might use (0.8, 0.1, 0.1) for non-office-home")

    # Read ggeur_data.py
    data_file = 'federatedscope/contrib/data/ggeur_data.py'
    if os.path.exists(data_file):
        with open(data_file, 'r', encoding='utf-8') as f:
            content = f.read()

        if 'seed=config.seed' in content:
            print("  Client uses: seed=config.seed")


def check_feature_cache_issue():
    """Check if feature cache might cause issues"""
    print("\n" + "=" * 80)
    print("Checking Feature Cache")
    print("=" * 80)

    cache_dir = 'clip_feature_cache'
    if os.path.exists(cache_dir):
        files = os.listdir(cache_dir)
        print(f"  Cache directory exists with {len(files)} files:")
        for f in files[:10]:
            print(f"    - {f}")
        if len(files) > 10:
            print(f"    ... and {len(files) - 10} more")

        # Check if cache includes split info
        has_split_info = any('split' in f for f in files)
        if has_split_info:
            print("  [OK] Cache filenames include split info")
        else:
            print("  WARNING: Cache filenames might not distinguish splits!")
    else:
        print(f"  Cache directory not found: {cache_dir}")


def simulate_full_training_data_flow():
    """Simulate the actual data flow during training"""
    print("\n" + "=" * 80)
    print("Simulating Full Data Flow")
    print("=" * 80)

    from federatedscope.cv.dataset.office_home import OfficeHome

    root = 'OfficeHomeDataset_10072016'
    if not os.path.exists(root):
        print(f"Dataset directory not found: {root}")
        return

    # Simulate config
    config_seed = 42
    config_splits = (0.7, 0.0, 0.3)

    print(f"\nConfig: seed={config_seed}, splits={config_splits}")

    all_client_train_paths = set()
    all_server_test_paths = set()

    for domain in OfficeHome.DOMAINS:
        # Client data loading (from ggeur_data.py)
        train_dataset = OfficeHome(
            root=root, domain=domain, split='train',
            transform=None, train_ratio=config_splits[0],
            val_ratio=config_splits[1], seed=config_seed
        )

        # Server test data loading (from ggeur_server.py)
        test_dataset = OfficeHome(
            root=root, domain=domain, split='test',
            transform=None, train_ratio=config_splits[0],
            val_ratio=config_splits[1], seed=config_seed
        )

        all_client_train_paths.update(train_dataset.data)
        all_server_test_paths.update(test_dataset.data)

        print(f"  {domain}: train={len(train_dataset)}, test={len(test_dataset)}")

    overlap = all_client_train_paths & all_server_test_paths

    print(f"\nTotal: train={len(all_client_train_paths)}, test={len(all_server_test_paths)}")
    print(f"Overlap: {len(overlap)}")

    if overlap:
        print("\n!!! CRITICAL: DATA LEAKAGE DETECTED !!!")
        overlap_ratio = len(overlap) / len(all_server_test_paths) * 100
        print(f"  {overlap_ratio:.1f}% of test data is in training set!")
    else:
        print("\n[OK] No data leakage in standard flow")


if __name__ == '__main__':
    print("\n" + "=" * 80)
    print("GGEUR_Clip Data Leakage Detection")
    print("=" * 80)

    # 1. Basic overlap check
    check_office_home_overlap()

    # 2. Different seeds check
    check_different_seeds()

    # 3. Server loading logic
    check_actual_server_loading()

    # 4. Feature cache check
    check_feature_cache_issue()

    # 5. Full flow simulation
    simulate_full_training_data_flow()

    print("\n" + "=" * 80)
    print("Detection Complete")
    print("=" * 80)
