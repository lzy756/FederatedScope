"""
GGEUR_Clip Step 4: Covariance Matrix Aggregation

This script:
1. Loads CLIP features from all clients for each class
2. Computes aggregated covariance matrix for each class
3. Saves covariance matrices for feature generation
"""

import numpy as np
import os
import argparse


# Client ID assignment
CLIENT_RANGE = {
    'Art': [0],
    'Clipart': [1],
    'Product': [2],
    'Real_World': [3]
}


def calculate_mean(Z):
    """Calculate mean of matrix Z"""
    return np.mean(Z, axis=0)


def calculate_covariance(Z, mean_Z):
    """Calculate covariance matrix of Z"""
    Z_centered = Z - mean_Z
    return (1 / Z.shape[0]) * np.dot(Z_centered.T, Z_centered)


def process_class_aggregated_covariance(class_idx, all_client_feature_paths):
    """
    Process all client features for a class and compute aggregated covariance matrix
    """
    sample_counts = []
    mean_matrices = []
    covariance_matrices = []

    for feature_file in all_client_feature_paths:
        if not os.path.exists(feature_file):
            continue

        Z = np.load(feature_file)

        # Skip if no samples
        if Z.size == 0 or Z.shape[0] == 0:
            continue

        mean_Z = calculate_mean(Z)
        cov_matrix = calculate_covariance(Z, mean_Z)
        sample_counts.append(Z.shape[0])
        mean_matrices.append(mean_Z)
        covariance_matrices.append(cov_matrix)

    # If no valid samples, return zero covariance matrix
    if len(sample_counts) == 0:
        print(f"  Class {class_idx}: No valid samples, returning zero matrix")
        return np.zeros((512, 512))

    # Compute aggregated mean and covariance matrix
    total_samples = np.sum(sample_counts)
    combined_mean = np.sum([sample_counts[i] * mean_matrices[i] for i in range(len(sample_counts))], axis=0) / total_samples

    # Aggregated covariance using the parallel axis theorem
    combined_cov_matrix = np.sum([sample_counts[i] * covariance_matrices[i] for i in range(len(sample_counts))], axis=0) / total_samples
    combined_cov_matrix += np.sum([sample_counts[i] * np.outer(mean_matrices[i] - combined_mean, mean_matrices[i] - combined_mean) for i in range(len(sample_counts))], axis=0) / total_samples

    print(f"  Class {class_idx}: {total_samples} total samples from {len(sample_counts)} clients")

    return combined_cov_matrix


def main(args):
    print("="*60)
    print("Computing Aggregated Covariance Matrices")
    print("="*60)

    base_dir = os.path.join(args.workspace, 'clip_train_features')
    output_dir = os.path.join(args.workspace, 'cov_matrices')
    os.makedirs(output_dir, exist_ok=True)

    num_classes = 65
    class_feature_data = []

    for class_idx in range(num_classes):
        all_client_feature_paths = []

        # Collect feature paths from all clients
        for dataset_name, clients in CLIENT_RANGE.items():
            for client_id in clients:
                feature_file = os.path.join(base_dir, dataset_name, f'client_{client_id}_class_{class_idx}_original_features.npy')
                all_client_feature_paths.append(feature_file)

        # Compute aggregated covariance matrix
        cov_matrix = process_class_aggregated_covariance(class_idx, all_client_feature_paths)
        class_feature_data.append(cov_matrix)

    # Save all covariance matrices
    for idx, cov_matrix in enumerate(class_feature_data):
        npy_path = os.path.join(output_dir, f'class_{idx}_cov_matrix.npy')
        np.save(npy_path, cov_matrix)

    print(f"\n{'='*60}")
    print(f"Saved {num_classes} covariance matrices to {output_dir}")
    print("="*60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='GGEUR_Clip Step 4: Covariance Matrix Aggregation')
    parser.add_argument('--workspace', type=str,
                        default='./ggeur_standalone/workspace',
                        help='Workspace directory')
    args = parser.parse_args()

    main(args)
