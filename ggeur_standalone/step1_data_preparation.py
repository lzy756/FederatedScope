"""
GGEUR_Clip Step 1: Data Preparation for Office-Home-LDS

This script:
1. Loads Office-Home dataset (Art, Clipart, Product, Real_World)
2. Splits data into 70% train / 30% test
3. Uses Dirichlet distribution (alpha=0.5) to allocate samples to 4 clients
4. Saves indices for later use
"""

import os
import numpy as np
from torchvision import transforms
from torchvision.datasets import ImageFolder
from tqdm import tqdm
from numpy.random import dirichlet
import shutil
import argparse


class ImageFolder_Custom(ImageFolder):
    """Custom ImageFolder that splits data into train/test"""
    def __init__(self, data_name, root, transform=None, target_transform=None, subset_train_ratio=0.7):
        super().__init__(os.path.join(root, data_name), transform=transform,
                         target_transform=target_transform)

        self.train_index_list = []
        self.test_index_list = []

        total_samples = len(self.samples)
        train_samples = int(subset_train_ratio * total_samples)

        shuffled_indices = np.random.permutation(total_samples)

        self.train_index_list = shuffled_indices[:train_samples].tolist()
        self.test_index_list = shuffled_indices[train_samples:].tolist()


def get_class_indices(dataset):
    """Get indices for each class in the dataset"""
    class_indices = {i: [] for i in range(65)}  # Office-Home has 65 classes
    for idx in range(len(dataset)):
        label = dataset.targets[idx]
        class_indices[label].append(idx)
    return class_indices


def generate_dirichlet_matrix(alpha, num_clients=4, num_classes=65):
    """Generate Dirichlet distribution matrix for data allocation"""
    return dirichlet([alpha] * num_clients, num_classes).T


def split_samples_for_domain(class_train_indices, dirichlet_column):
    """Split samples to client based on Dirichlet proportions"""
    client_indices = []
    class_proportions = {}

    for class_label, indices in class_train_indices.items():
        num_samples = len(indices)
        if num_samples == 0:
            continue

        proportion = dirichlet_column[class_label]
        class_proportions[class_label] = proportion

        num_to_allocate = int(proportion * num_samples)
        allocated_indices = indices[:num_to_allocate]
        client_indices.extend(allocated_indices)

    return client_indices, class_proportions


def save_indices(indices_dict, domain_name, file_type, output_dir):
    """Save indices to files"""
    domain_dir = os.path.join(output_dir, domain_name)
    os.makedirs(domain_dir, exist_ok=True)

    for key, indices in indices_dict.items():
        npy_filename = os.path.join(domain_dir, f"{file_type}_{key}_indices.npy")
        np.save(npy_filename, np.array(indices))
        print(f"Saved {file_type}_{key} indices: {len(indices)} samples")


def save_class_indices(class_indices, domain_name, output_dir):
    """Save class indices"""
    domain_dir = os.path.join(output_dir, domain_name)
    os.makedirs(domain_dir, exist_ok=True)

    npy_filename = os.path.join(domain_dir, 'class_indices.npy')
    np.save(npy_filename, class_indices)
    print(f"Saved class indices for {domain_name}")


def get_domain_folder_name(domain, data_path):
    """Get actual folder name for domain (handles 'Real World' vs 'Real_World')"""
    # Try the domain name as-is first
    if os.path.exists(os.path.join(data_path, domain)):
        return domain
    # Try with space instead of underscore
    domain_with_space = domain.replace('_', ' ')
    if os.path.exists(os.path.join(data_path, domain_with_space)):
        return domain_with_space
    # Try with underscore instead of space
    domain_with_underscore = domain.replace(' ', '_')
    if os.path.exists(os.path.join(data_path, domain_with_underscore)):
        return domain_with_underscore
    return domain


def main(args):
    np.random.seed(args.seed)

    output_dir = os.path.join(args.output_base, 'output_indices')
    os.makedirs(output_dir, exist_ok=True)

    # Domain names for output (standardized)
    domains = ['Art', 'Clipart', 'Product', 'Real_World']

    # Generate Dirichlet matrix: 4 clients x 65 classes
    dirichlet_matrix = generate_dirichlet_matrix(args.alpha)
    print(f"Generated 4x65 Dirichlet matrix with alpha={args.alpha}")

    all_class_proportions = {}

    for domain_index, domain in enumerate(domains):
        print(f"\n{'='*50}")
        print(f"Processing domain {domain_index}: {domain}")
        print(f"{'='*50}")

        # Get actual folder name
        folder_name = get_domain_folder_name(domain, args.data_path)
        print(f"Using folder: {folder_name}")

        dataset = ImageFolder_Custom(
            data_name=folder_name,
            root=args.data_path,
            transform=transforms.ToTensor()
        )

        # Get and save class indices
        class_indices = get_class_indices(dataset)
        save_class_indices(class_indices, domain, output_dir)

        # Get train/test indices
        train_indices = dataset.train_index_list
        test_indices = dataset.test_index_list

        print(f"Total samples: {len(dataset)}")
        print(f"Train samples: {len(train_indices)}")
        print(f"Test samples: {len(test_indices)}")

        save_indices({'train': train_indices}, domain, file_type="train", output_dir=output_dir)
        save_indices({'test': test_indices}, domain, file_type="test", output_dir=output_dir)

        # Get class-wise train indices
        class_train_indices = {class_label: [] for class_label in class_indices.keys()}
        for idx in train_indices:
            label = dataset.targets[idx]
            class_train_indices[label].append(idx)

        # Allocate samples to client using Dirichlet
        client_indices, class_proportions = split_samples_for_domain(
            class_train_indices,
            dirichlet_matrix[domain_index]
        )

        all_class_proportions[domain] = class_proportions
        save_indices({'client': client_indices}, domain, file_type="client", output_dir=output_dir)
        print(f"Client samples (after Dirichlet): {len(client_indices)}")

    print(f"\n{'='*50}")
    print("Data preparation completed!")
    print(f"Output directory: {output_dir}")
    print(f"{'='*50}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GGEUR_Clip Step 1: Data Preparation')
    parser.add_argument('--data_path', type=str,
                        default='/root/CDA_new/OfficeHomeDataset_10072016',
                        help='Path to Office-Home dataset')
    parser.add_argument('--output_base', type=str,
                        default='./ggeur_standalone/workspace',
                        help='Base output directory')
    parser.add_argument('--alpha', type=float, default=0.1,
                        help='Dirichlet concentration parameter')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    args = parser.parse_args()

    main(args)
