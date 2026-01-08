"""
GGEUR_Clip Step 5: Prototype Computation

This script:
1. Loads CLIP features for each client and class
2. Computes prototype (mean feature) for each client-class pair
3. Saves prototypes for use in feature generation
"""

import os
import numpy as np
import argparse


# Client ID assignment
CLIENT_RANGE = {
    'Art': [0],
    'Clipart': [1],
    'Product': [2],
    'Real_World': [3]
}


def compute_prototype(features):
    """Compute prototype (mean) from features"""
    return np.mean(features, axis=0)


def process_prototypes(dataset_name, client_id, base_dir, output_dir, num_classes=65):
    """Compute and save prototype for each class of a client"""
    prototypes_saved = 0

    for class_label in range(num_classes):
        feature_file = os.path.join(base_dir, dataset_name, f'client_{client_id}_class_{class_label}_original_features.npy')

        if not os.path.exists(feature_file):
            continue

        features = np.load(feature_file)

        # Skip if empty
        if features.size == 0 or features.shape[0] == 0:
            continue

        # Compute prototype
        prototype = compute_prototype(features)

        # Save prototype
        prototype_file = os.path.join(output_dir, f'client_{client_id}_class_{class_label}_prototype.npy')
        np.save(prototype_file, prototype)
        prototypes_saved += 1

    print(f"  {dataset_name} client {client_id}: {prototypes_saved} prototypes saved")


def main(args):
    print("="*60)
    print("Computing Prototypes")
    print("="*60)

    base_dir = os.path.join(args.workspace, 'clip_train_features')
    output_base_dir = os.path.join(args.workspace, 'prototypes')
    os.makedirs(output_base_dir, exist_ok=True)

    domains = ['Art', 'Clipart', 'Product', 'Real_World']

    for dataset_name in domains:
        print(f"\nProcessing {dataset_name}...")

        assigned_clients = CLIENT_RANGE[dataset_name]
        domain_output_dir = os.path.join(output_base_dir, dataset_name)
        os.makedirs(domain_output_dir, exist_ok=True)

        for client_id in assigned_clients:
            process_prototypes(dataset_name, client_id, base_dir, domain_output_dir)

    print(f"\n{'='*60}")
    print(f"Prototypes saved to {output_base_dir}")
    print("="*60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='GGEUR_Clip Step 5: Prototype Computation')
    parser.add_argument('--workspace', type=str,
                        default='./ggeur_standalone/workspace',
                        help='Workspace directory')
    args = parser.parse_args()

    main(args)
