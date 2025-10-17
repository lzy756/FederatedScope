"""
Test script for Office-Caltech dataset loader

Usage:
    python scripts/test_office_caltech_loader.py --data_dir /home/liziyu/data/office_caltech_10
"""

import sys
import os
import argparse

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from federatedscope.cv.dataset.office_caltech import OfficeCaltech, load_office_caltech_domain_data
from torchvision import transforms
import torch


def test_single_domain(data_dir, domain='amazon'):
    """Test loading a single domain"""
    print(f"\n{'='*60}")
    print(f"Testing domain: {domain}")
    print(f"{'='*60}")

    # Define transform
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])

    try:
        # Load domain data
        domain_data = load_office_caltech_domain_data(
            root=data_dir,
            domain=domain,
            splits=(0.8, 0.1, 0.1),
            transform=transform,
            seed=123
        )

        print(f"\n✓ Successfully loaded {domain} domain!")
        print(f"  Train samples: {len(domain_data['train'])}")
        print(f"  Val samples:   {len(domain_data['val'])}")
        print(f"  Test samples:  {len(domain_data['test'])}")

        # Test loading a sample
        if len(domain_data['train']) > 0:
            sample_img, sample_label = domain_data['train'][0]
            print(f"\n  Sample image shape: {sample_img.shape}")
            print(f"  Sample label: {sample_label} ({OfficeCaltech.CLASSES[sample_label]})")

            # Check class distribution
            labels = domain_data['train'].targets
            unique_labels = set(labels)
            print(f"\n  Classes present: {len(unique_labels)} out of {len(OfficeCaltech.CLASSES)}")
            for label_id in sorted(unique_labels):
                count = labels.count(label_id)
                print(f"    {OfficeCaltech.CLASSES[label_id]}: {count} images")

        return True

    except Exception as e:
        print(f"\n✗ Error loading {domain} domain:")
        print(f"  {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_all_domains(data_dir):
    """Test loading all domains"""
    print("\n" + "="*60)
    print("Testing Office-Caltech Dataset Loader")
    print("="*60)
    print(f"\nData directory: {data_dir}")

    if not os.path.exists(data_dir):
        print(f"\n✗ Error: Data directory does not exist!")
        print(f"  Please ensure the dataset is located at: {data_dir}")
        return False

    # Check directory structure
    print(f"\nChecking directory structure...")
    domains = ['amazon', 'webcam', 'dslr', 'caltech']
    missing_domains = []

    for domain in domains:
        domain_path = os.path.join(data_dir, domain)
        if os.path.exists(domain_path):
            num_classes = len([d for d in os.listdir(domain_path)
                             if os.path.isdir(os.path.join(domain_path, d))])
            print(f"  ✓ {domain}: {num_classes} class folders")
        else:
            print(f"  ✗ {domain}: NOT FOUND")
            missing_domains.append(domain)

    if missing_domains:
        print(f"\n✗ Missing domains: {missing_domains}")
        print(f"  Expected structure:")
        print(f"    {data_dir}/")
        for domain in domains:
            print(f"      ├── {domain}/")
        return False

    # Test loading each domain
    success_count = 0
    for domain in domains:
        if test_single_domain(data_dir, domain):
            success_count += 1

    # Summary
    print(f"\n{'='*60}")
    print(f"Summary: {success_count}/{len(domains)} domains loaded successfully")
    print(f"{'='*60}")

    if success_count == len(domains):
        print("\n✅ All tests passed! Dataset is ready to use.")
        print("\nYou can now run FedLSA with:")
        print("  python scripts/run_fedlsa.py --cfg scripts/example_configs/fedlsa_office_caltech.yaml")
        return True
    else:
        print("\n⚠️  Some domains failed to load. Please check the errors above.")
        return False


def main():
    parser = argparse.ArgumentParser(description='Test Office-Caltech dataset loader')
    parser.add_argument('--data_dir', type=str,
                       default='/home/liziyu/data/office_caltech_10',
                       help='Path to Office-Caltech dataset')
    args = parser.parse_args()

    success = test_all_domains(args.data_dir)
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()

