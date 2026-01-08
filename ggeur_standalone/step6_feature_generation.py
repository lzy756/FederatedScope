"""
GGEUR_Clip Step 6: Feature Generation (GGEUR_Clip Core)

This is the core GGEUR_Clip algorithm that:
1. Expands local features using aggregated covariance matrices
2. Generates features from other clients' prototypes
3. Combines original and generated features to create balanced datasets

Key parameters:
- target_size: Target number of samples per class (default 50)
- num_per_sample: Number of samples to generate per original sample (default 50)
"""

import os
import numpy as np
import argparse
from tqdm import tqdm


# Client ID assignment
DATASETS_CONFIG = {
    'Art': {'client_ids': [0], 'num_generated_per_sample': 50},
    'Clipart': {'client_ids': [1], 'num_generated_per_sample': 50},
    'Product': {'client_ids': [2], 'num_generated_per_sample': 50},
    'Real_World': {'client_ids': [3], 'num_generated_per_sample': 50}
}


def nearest_pos_def(cov_matrix):
    """
    Ensure covariance matrix is positive definite
    Uses eigenvalue scaling to make the matrix well-conditioned
    """
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

    # Scale first 10 eigenvalues for better conditioning
    scale_factors = np.ones_like(eigenvalues)
    scale_factors[:10] = np.linspace(5, 1, 10)
    eigenvalues = eigenvalues * scale_factors

    # Clip negative eigenvalues
    eigenvalues[eigenvalues < 0] = 0

    return eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T


def generate_new_samples(feature, cov_matrix, num_generated):
    """
    Generate new samples using Gaussian distribution
    centered at the given feature with the given covariance matrix
    """
    cov_matrix = nearest_pos_def(cov_matrix)

    # Add small jitter for numerical stability
    jitter = 1e-6
    while True:
        try:
            B = np.linalg.cholesky(cov_matrix + jitter * np.eye(cov_matrix.shape[0]))
            break
        except np.linalg.LinAlgError:
            jitter *= 10
            if jitter > 1:
                # Fallback to diagonal covariance
                B = np.eye(cov_matrix.shape[0]) * 0.1
                break

    new_features = np.random.multivariate_normal(feature, B @ B.T, num_generated)
    return new_features


def expand_features_with_cov(original_features, cov_matrix, num_per_sample, target_select=50):
    """
    Expand features by generating new samples from each original feature
    Returns a random selection of generated samples
    """
    generated_samples = []

    for feature in original_features:
        new_samples = generate_new_samples(feature, cov_matrix, num_per_sample)
        generated_samples.append(new_samples)

    all_generated_samples = np.vstack(generated_samples)

    # Randomly select target_select samples
    if all_generated_samples.shape[0] >= target_select:
        selected_indices = np.random.choice(all_generated_samples.shape[0], target_select, replace=False)
        return all_generated_samples[selected_indices]
    else:
        return all_generated_samples


def combine_samples(original_features, expanded_features, other_generated_samples, target_size=50):
    """
    Combine original samples with generated samples to reach target_size
    Priority: original samples > expanded samples > other client samples
    """
    num_original = original_features.shape[0]

    # If original samples exceed target, subsample
    if num_original >= target_size:
        selected_indices = np.random.choice(num_original, target_size, replace=False)
        return original_features[selected_indices]

    # Need to supplement with generated samples
    num_needed = target_size - num_original

    # Combine all generated samples
    combined_generated_samples = []
    if expanded_features.shape[0] > 0:
        combined_generated_samples.append(expanded_features)
    if other_generated_samples.shape[0] > 0:
        combined_generated_samples.append(other_generated_samples)

    if len(combined_generated_samples) > 0:
        combined_generated_samples = np.vstack(combined_generated_samples)

        if combined_generated_samples.shape[0] >= num_needed:
            selected_indices = np.random.choice(combined_generated_samples.shape[0], num_needed, replace=False)
            final_samples = np.vstack((original_features, combined_generated_samples[selected_indices]))
        else:
            # Not enough generated samples, use all
            final_samples = np.vstack((original_features, combined_generated_samples))
    else:
        final_samples = original_features

    return final_samples


def process_clients(client_id, class_idx, original_features, cov_matrices, prototype_features, num_per_sample, target_size=50):
    """
    Process a client's class data:
    1. If has original features: expand them using covariance matrix
    2. Generate features from other clients' prototypes
    3. Combine all samples
    """
    final_samples = None
    feature_dim = cov_matrices[class_idx].shape[0]

    # 1. Expand original features if available
    if original_features.shape[0] > 0:
        expanded_features = expand_features_with_cov(original_features, cov_matrices[class_idx], num_per_sample)
    else:
        expanded_features = np.empty((0, feature_dim))

    # 2. Generate samples from other clients' prototypes
    additional_samples = []
    for prototype in prototype_features:
        if prototype.size > 0:
            new_samples = generate_new_samples(prototype, cov_matrices[class_idx], 50)
            additional_samples.append(new_samples)

    if len(additional_samples) > 0:
        additional_samples = np.vstack(additional_samples)
    else:
        additional_samples = np.empty((0, feature_dim))

    # 3. Combine samples
    if original_features.shape[0] == 0:
        # No original features, use only generated from prototypes
        final_samples = additional_samples[:target_size] if additional_samples.shape[0] >= target_size else additional_samples
    else:
        final_samples = combine_samples(original_features, expanded_features, additional_samples, target_size=target_size)

    return final_samples


def main(args):
    np.random.seed(args.seed)

    print("="*60)
    print("GGEUR_Clip Feature Generation")
    print("="*60)

    base_dir = os.path.join(args.workspace, 'clip_train_features')
    cov_dir = os.path.join(args.workspace, 'cov_matrices')
    prototype_dir = os.path.join(args.workspace, 'prototypes')
    output_dir = os.path.join(args.workspace, 'augmented_features')
    os.makedirs(output_dir, exist_ok=True)

    # Load covariance matrices
    print("\nLoading covariance matrices...")
    cov_matrices = {}
    for class_idx in range(65):
        cov_path = os.path.join(cov_dir, f'class_{class_idx}_cov_matrix.npy')
        if os.path.exists(cov_path):
            cov_matrices[class_idx] = np.load(cov_path)
        else:
            print(f"Warning: Covariance matrix for class {class_idx} not found")
            cov_matrices[class_idx] = np.eye(512) * 0.01

    # Process each dataset
    for dataset_name, config in DATASETS_CONFIG.items():
        print(f"\n{'='*50}")
        print(f"Processing {dataset_name}")
        print(f"{'='*50}")

        for client_id in config['client_ids']:
            dataset_output_dir = os.path.join(output_dir, dataset_name)
            os.makedirs(dataset_output_dir, exist_ok=True)

            total_samples = 0

            for class_idx in tqdm(range(65), desc=f"Client {client_id}"):
                # Load original features
                original_features_path = os.path.join(base_dir, dataset_name, f'client_{client_id}_class_{class_idx}_original_features.npy')

                if os.path.exists(original_features_path):
                    original_features = np.load(original_features_path)
                    if original_features.size == 0:
                        original_features = np.empty((0, 512))
                else:
                    original_features = np.empty((0, 512))

                # Load prototypes from other datasets
                prototype_features = []
                other_datasets = [ds for ds in DATASETS_CONFIG.keys() if ds != dataset_name]

                for other_dataset in other_datasets:
                    other_client_id = DATASETS_CONFIG[other_dataset]['client_ids'][0]
                    prototype_path = os.path.join(prototype_dir, other_dataset, f'client_{other_client_id}_class_{class_idx}_prototype.npy')

                    if os.path.exists(prototype_path):
                        prototype = np.load(prototype_path)
                        prototype_features.append(prototype)

                # Generate augmented features
                final_samples = process_clients(
                    client_id, class_idx, original_features,
                    cov_matrices, prototype_features,
                    config['num_generated_per_sample'],
                    target_size=args.target_size
                )

                # Save augmented features
                if final_samples is not None and final_samples.shape[0] > 0:
                    client_class_dir = os.path.join(dataset_output_dir, f'client_{client_id}_class_{class_idx}')
                    os.makedirs(client_class_dir, exist_ok=True)

                    np.save(os.path.join(client_class_dir, 'final_embeddings_filled.npy'), final_samples)

                    labels = np.full(final_samples.shape[0], class_idx)
                    np.save(os.path.join(client_class_dir, 'labels_filled.npy'), labels)

                    total_samples += final_samples.shape[0]

            print(f"  Client {client_id}: Total {total_samples} samples across all classes")

    print(f"\n{'='*60}")
    print(f"Feature generation completed!")
    print(f"Output directory: {output_dir}")
    print("="*60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='GGEUR_Clip Step 6: Feature Generation')
    parser.add_argument('--workspace', type=str,
                        default='./ggeur_standalone/workspace',
                        help='Workspace directory')
    parser.add_argument('--target_size', type=int, default=50,
                        help='Target number of samples per class')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    args = parser.parse_args()

    main(args)
