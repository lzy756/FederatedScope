"""
GGEUR_Clip Step 2: Client-Class Index Matching

This script:
1. Matches client indices with class indices
2. Generates per-client per-class index files for feature extraction
"""

import os
import numpy as np
import argparse


# Client ID assignment: each domain maps to one client
CLIENT_RANGE = {
    'Art': [0],
    'Clipart': [1],
    'Product': [2],
    'Real_World': [3]
}


def load_combined_class_indices(class_indices_file):
    """Load combined class indices file"""
    return np.load(class_indices_file, allow_pickle=True).item()


def load_client_indices(client_indices_path):
    """Load client indices file"""
    return np.load(client_indices_path)


def match_client_class_indices(client_indices, class_indices):
    """Match client indices with class indices to generate per-class indices"""
    client_class_indices = {class_label: [] for class_label in class_indices.keys()}

    for index in client_indices:
        for class_label, class_idx_list in class_indices.items():
            if index in class_idx_list:
                client_class_indices[class_label].append(index)
                break

    return client_class_indices


def save_client_class_indices(client_class_indices, output_dir, client_id):
    """Save per-client per-class indices"""
    os.makedirs(output_dir, exist_ok=True)

    for class_label, indices in client_class_indices.items():
        npy_path = os.path.join(output_dir, f'client_{client_id}_class_{class_label}_indices.npy')
        np.save(npy_path, indices)

        if len(indices) > 0:
            print(f"Client {client_id} Class {class_label}: {len(indices)} samples")


def main(args):
    input_dir = os.path.join(args.workspace, 'output_indices')
    output_dir = os.path.join(args.workspace, 'output_client_class_indices')
    os.makedirs(output_dir, exist_ok=True)

    domains = ['Art', 'Clipart', 'Product', 'Real_World']

    for dataset_name in domains:
        print(f"\n{'='*50}")
        print(f"Processing {dataset_name}")
        print(f"{'='*50}")

        # Load combined class indices
        combined_class_indices_file = os.path.join(input_dir, dataset_name, 'class_indices.npy')
        if not os.path.exists(combined_class_indices_file):
            print(f"Error: {combined_class_indices_file} not found. Run step1 first.")
            continue

        class_indices = load_combined_class_indices(combined_class_indices_file)

        # Process each client
        assigned_clients = CLIENT_RANGE[dataset_name]

        for client_id in assigned_clients:
            client_indices_path = os.path.join(input_dir, dataset_name, 'client_client_indices.npy')

            if not os.path.exists(client_indices_path):
                print(f"Warning: {client_indices_path} not found, skipping...")
                continue

            client_indices = load_client_indices(client_indices_path)

            # Match client indices to classes
            client_class_indices = match_client_class_indices(client_indices, class_indices)

            # Save results
            domain_output_dir = os.path.join(output_dir, dataset_name)
            save_client_class_indices(client_class_indices, domain_output_dir, client_id)

    print(f"\n{'='*50}")
    print("Client-class index matching completed!")
    print(f"Output directory: {output_dir}")
    print(f"{'='*50}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GGEUR_Clip Step 2: Client-Class Index Matching')
    parser.add_argument('--workspace', type=str,
                        default='./ggeur_standalone/workspace',
                        help='Workspace directory (output of step1)')
    args = parser.parse_args()

    main(args)
