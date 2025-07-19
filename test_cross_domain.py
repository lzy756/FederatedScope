#!/usr/bin/env python3

import sys
import os
sys.path.append('/home/liruobin/FL/FederatedScope')

from federatedscope.core.configs.config import global_cfg
from federatedscope.contrib.data.digits import load_cross_domain_data

def test_cross_domain_data():
    """Test the cross domain data loading and merged distribution calculation"""
    
    # Create a minimal config
    config = global_cfg.clone()
    config.data.root = 'data/'
    config.data.type = 'cross_domain'
    config.data.splits = [0.6, 0.2, 0.2]
    config.data.subsample = 0.1  # Use smaller subset for testing
    config.data.transform = [['ToTensor'], ['Normalize', {'mean': [0.5], 'std': [0.5]}]]
    config.data.splitter = 'iid'
    config.federate.client_num = 100
    
    print("Testing cross domain data loading...")
    print(f"Config: {config.data}")
    
    try:
        # Test the data loading
        data_dict, modified_config = load_cross_domain_data(config, client_cfgs=None)
        
        print(f"\nSuccess! Data loaded with {len(data_dict)} entries")
        print(f"Keys: {sorted(list(data_dict.keys())[:10])}...")  # Show first 10 keys
        
        # Check if files were created
        files_to_check = [
            'data_info.txt',
            'test_data_info.txt', 
            'peer_communities.txt',
            'merged_distribution.txt'
        ]
        
        print("\nChecking generated files:")
        for filename in files_to_check:
            if os.path.exists(filename):
                print(f"✓ {filename} created successfully")
                # Show first few lines
                with open(filename, 'r') as f:
                    lines = f.readlines()[:5]
                    print(f"  Preview: {lines[0].strip()}")
            else:
                print(f"✗ {filename} not found")
        
        return True
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_cross_domain_data()
    if success:
        print("\n✓ Test completed successfully!")
    else:
        print("\n✗ Test failed!")
