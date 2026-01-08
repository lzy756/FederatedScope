"""Diagnose PACS data loading issue"""
import sys
sys.path.insert(0, '.')

from federatedscope.core.configs import global_cfg
from federatedscope.core.auxiliaries.data_builder import get_data

# Load config
cfg = global_cfg.clone()
cfg.merge_from_file('scripts/example_configs/fedavg_pacs_domain_eval.yaml')

print("="*80)
print("Testing PACS data loading...")
print("="*80)

# Try to load data
try:
    data, modified_cfg = get_data(config=cfg)
    print('\n✓ Data loaded successfully')
    print(f'Data type: {type(data)}')

    if hasattr(data, 'keys'):
        print(f'Number of clients: {len(data)}')
        print(f'Client IDs: {list(data.keys())[:10]}...')

        first_client = list(data.keys())[0]
        print(f'\nFirst client ID: {first_client}')
        print(f'First client data type: {type(data[first_client])}')

        if hasattr(data[first_client], 'keys'):
            print(f'First client data keys: {list(data[first_client].keys())}')

            for split in ['train', 'val', 'test']:
                if split in data[first_client]:
                    split_data = data[first_client][split]
                    print(f'\n{split} data:')
                    print(f'  Type: {type(split_data)}')
                    print(f'  Length: {len(split_data)}')
                    if hasattr(split_data, 'dataset'):
                        print(f'  Is Subset: True')
                        print(f'  Base dataset type: {type(split_data.dataset)}')

        elif hasattr(data[first_client], '__dict__'):
            print(f'First client data attributes: {list(data[first_client].__dict__.keys())}')
    else:
        print(f'Data is not a dict. Type: {type(data)}')

except Exception as e:
    print(f'\n✗ Error: {e}')
    import traceback
    traceback.print_exc()

print("\n" + "="*80)
