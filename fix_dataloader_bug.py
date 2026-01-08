#!/usr/bin/env python
"""
Fix DataLoader vs Dataset bug in GGEUR_Clip implementation.

This script fixes the issue where DataLoader objects are passed to
extract_dataset_features() which expects Dataset objects.
"""

import os
import sys

def fix_ggeur_client():
    """Fix federatedscope/contrib/worker/ggeur_client.py"""
    filepath = "federatedscope/contrib/worker/ggeur_client.py"

    if not os.path.exists(filepath):
        print(f"Error: {filepath} not found")
        return False

    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    # Fix line 121
    old_line = "        train_data = self.data['train']"
    new_line = "        # self.data['train'] is a DataLoader, we need the underlying Dataset\n        train_data = self.data['train'].dataset"

    if old_line in content:
        content = content.replace(old_line, new_line)
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"✓ Fixed {filepath}")
        return True
    else:
        print(f"⚠ Pattern not found in {filepath}, may already be fixed")
        return True

def fix_ggeur_trainer():
    """Fix federatedscope/contrib/trainer/ggeur_trainer.py"""
    filepath = "federatedscope/contrib/trainer/ggeur_trainer.py"

    if not os.path.exists(filepath):
        print(f"Error: {filepath} not found")
        return False

    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    # Fix line 216
    old_line = "        test_data = self.data[target_data_split_name]"
    new_line = "        # self.data is a dict of DataLoaders, we need the underlying Dataset\n        test_data = self.data[target_data_split_name].dataset"

    if old_line in content:
        content = content.replace(old_line, new_line)
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"✓ Fixed {filepath}")
        return True
    else:
        print(f"⚠ Pattern not found in {filepath}, may already be fixed")
        return True

def main():
    print("Fixing DataLoader vs Dataset bug in GGEUR_Clip implementation...\n")

    success1 = fix_ggeur_client()
    success2 = fix_ggeur_trainer()

    if success1 and success2:
        print("\n✓ All fixes applied successfully!")
        print("\nYou can now run your GGEUR_Clip experiment again:")
        print("  python run_ggeur.py --cfg scripts/example_configs/ggeur_officehome_lds.yaml")
        return 0
    else:
        print("\n✗ Some fixes failed")
        return 1

if __name__ == '__main__':
    sys.exit(main())
