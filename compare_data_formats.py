"""Compare PACS and office_caltech data loading formats"""
import sys
sys.path.insert(0, '.')

from federatedscope.cv.dataloader import load_cv_dataset
from federatedscope.core.configs import global_cfg

def test_dataset(dataset_type, config_file):
    print(f"\n{'='*80}")
    print(f"Testing {dataset_type.upper()} dataset")
    print(f"{'='*80}")

    # Load config
    cfg = global_cfg.clone()
    cfg.merge_from_file(config_file)

    # Load dataset
    try:
        data_dict, modified_cfg = load_cv_dataset(cfg)

        print(f"\n✓ Data loaded successfully")
        print(f"Number of clients: {len(data_dict)}")
        print(f"Client IDs (first 10): {list(data_dict.keys())[:10]}")

        # Check first client
        first_client_id = list(data_dict.keys())[0]
        client_data = data_dict[first_client_id]

        print(f"\nFirst client (ID={first_client_id}):")
        print(f"  Type: {type(client_data)}")

        if isinstance(client_data, dict):
            print(f"  Keys: {list(client_data.keys())}")

            for split_name in ['train', 'val', 'test']:
                if split_name in client_data:
                    split_data = client_data[split_name]
                    print(f"\n  {split_name}:")
                    print(f"    Type: {type(split_data)}")
                    print(f"    Length: {len(split_data)}")

                    # Check if it's a Subset
                    if hasattr(split_data, 'dataset'):
                        print(f"    Is Subset: True")
                        print(f"    Dataset type: {type(split_data.dataset)}")
                        print(f"    Has targets: {hasattr(split_data.dataset, 'targets')}")
                    elif hasattr(split_data, 'targets'):
                        print(f"    Has targets: True")
                        print(f"    Targets length: {len(split_data.targets)}")
        else:
            print(f"  WARNING: client_data is not a dict!")
            print(f"  Attributes: {dir(client_data)}")

        return True

    except Exception as e:
        print(f"\n✗ Error loading {dataset_type}: {e}")
        import traceback
        traceback.print_exc()
        return False

# Test both datasets
print("\n" + "="*80)
print("COMPARING DATA FORMATS")
print("="*80)

# Test PACS
pacs_success = test_dataset('pacs', 'scripts/example_configs/fedavg_pacs_domain_eval.yaml')

# Test office_caltech (if available)
# office_success = test_dataset('office_caltech', 'scripts/example_configs/fedavg_office_caltech_domain_eval.yaml')

print("\n" + "="*80)
print("SUMMARY")
print("="*80)
print(f"PACS: {'✓ SUCCESS' if pacs_success else '✗ FAILED'}")
# print(f"Office-Caltech: {'✓ SUCCESS' if office_success else '✗ FAILED'}")
