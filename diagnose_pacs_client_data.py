"""
Diagnose PACS client data loading issue
"""
import os
import sys
import numpy as np

# Add FederatedScope to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from federatedscope.core.configs.config import global_cfg
from federatedscope.core.auxiliaries.utils import setup_seed
from federatedscope.cv.dataloader.dataloader import load_cv_dataset

def main():
    # Initialize config
    cfg = global_cfg.clone()

    # Load PACS config
    cfg.merge_from_file('scripts/example_configs/cross_domain_adaptive_pacs.yaml')

    # Override for local testing
    cfg.merge_from_list([
        'data.root', 'D:/Projects/FederatedScope/data/PACS',
    ])

    setup_seed(cfg.seed)

    print("="*80)
    print("Configuration:")
    print(f"  Dataset: {cfg.data.type}")
    print(f"  Client num: {cfg.federate.client_num}")
    print(f"  Sample client num: {cfg.federate.sample_client_num}")
    print(f"  Splits: {cfg.data.splits}")
    print(f"  Server test samples per class: {cfg.data.server_test_samples_per_class}")
    print(f"  Dirichlet alpha: {cfg.data.dirichlet_alpha}")
    print(f"  Make global eval: {cfg.federate.make_global_eval}")
    print("="*80)

    # Load dataset
    print("\nLoading dataset...")
    try:
        data_dict, config = load_cv_dataset(cfg)

        print(f"\n✓ Dataset loaded successfully")
        print(f"  Total clients: {len(data_dict)}")

        # Check each client's data
        empty_clients = []
        for client_id in sorted(data_dict.keys()):
            client_data = data_dict[client_id]

            train_size = len(client_data.get('train', [])) if 'train' in client_data else 0
            val_size = len(client_data.get('val', [])) if 'val' in client_data else 0
            test_size = len(client_data.get('test', [])) if 'test' in client_data else 0

            if train_size == 0:
                empty_clients.append(client_id)

            if client_id <= 10 or train_size == 0:  # Show first 10 and any empty clients
                print(f"\n  Client {client_id}:")
                print(f"    Train: {train_size}")
                print(f"    Val: {val_size}")
                print(f"    Test: {test_size}")

                # Show class distribution for train set
                if train_size > 0 and hasattr(client_data['train'], 'targets'):
                    targets = np.array(client_data['train'].targets)
                    unique, counts = np.unique(targets, return_counts=True)
                    class_dist = dict(zip(unique.tolist(), counts.tolist()))
                    print(f"    Class distribution: {class_dist}")
                elif train_size > 0:
                    # Handle Subset case
                    try:
                        dataset = client_data['train']
                        if hasattr(dataset, 'dataset'):
                            targets = np.array(dataset.dataset.targets)[dataset.indices]
                            unique, counts = np.unique(targets, return_counts=True)
                            class_dist = dict(zip(unique.tolist(), counts.tolist()))
                            print(f"    Class distribution: {class_dist}")
                    except:
                        pass

        if empty_clients:
            print(f"\n⚠ WARNING: {len(empty_clients)} clients have NO training data!")
            print(f"  Empty client IDs: {empty_clients[:20]}{'...' if len(empty_clients) > 20 else ''}")
        else:
            print(f"\n✓ All clients have training data")

    except Exception as e:
        print(f"\n✗ Error loading dataset: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()
