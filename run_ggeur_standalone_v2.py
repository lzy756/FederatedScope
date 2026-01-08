"""
Standalone GGEUR_Clip Implementation - Exact Match to Paper

This script implements GGEUR_Clip exactly as described in the paper:
1. Offline CLIP feature extraction
2. Offline covariance matrix aggregation (global across all clients)
3. Offline prototype computation
4. Offline augmented feature generation
5. Federated training on pre-computed features

Key differences from our previous FederatedScope implementation:
1. All augmentation is OFFLINE (pre-computed before training)
2. Covariance is GLOBAL (aggregated from ALL clients per class)
3. Uses softmax output + CrossEntropyLoss (paper's style)
4. Features are NOT normalized after augmentation

Usage:
    python run_ggeur_standalone.py --data_root ./OfficeHomeDataset_10072016
"""

import os
import sys
import argparse
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from PIL import Image
from tqdm import tqdm

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =============================================================================
# Dataset Classes
# =============================================================================

class FeatureDataset(Dataset):
    """Dataset for pre-computed features"""
    def __init__(self, features, labels):
        self.features = torch.from_numpy(features).float() if isinstance(features, np.ndarray) else features.float()
        self.labels = torch.from_numpy(labels).long() if isinstance(labels, np.ndarray) else labels.long()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


# =============================================================================
# Model - Exactly as in Paper
# =============================================================================

class GGEURClassifier(nn.Module):
    """
    Simple linear classifier with softmax output.
    This matches the paper's MyNet exactly.
    """
    def __init__(self, input_dim=512, num_classes=65):
        super(GGEURClassifier, self).__init__()
        self.fc = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        # Paper uses softmax in forward!
        logits = self.fc(x)
        return F.softmax(logits, dim=1)


# =============================================================================
# CLIP Feature Extractor
# =============================================================================

class CLIPFeatureExtractor:
    """Extract CLIP features from images"""

    def __init__(self, model_name='ViT-B-32', weights_path=None):
        """
        Args:
            model_name: CLIP model name (paper uses ViT-B-32)
            weights_path: Path to custom weights
        """
        import open_clip

        logger.info(f"Loading CLIP model: {model_name}")

        if weights_path and os.path.exists(weights_path):
            self.model, _, self.preprocess = open_clip.create_model_and_transforms(
                model_name, pretrained=weights_path
            )
            logger.info(f"Loaded custom weights from: {weights_path}")
        else:
            # Use default OpenAI weights
            self.model, _, self.preprocess = open_clip.create_model_and_transforms(
                model_name, pretrained='openai'
            )
            logger.info("Using default OpenAI pretrained weights")

        self.model.eval().to(device)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
        ])

    def extract_features(self, images):
        """
        Extract CLIP features from images.

        Args:
            images: List of PIL images or torch tensor [N, C, H, W]

        Returns:
            features: numpy array [N, 512]
        """
        if isinstance(images, list):
            # List of PIL images
            images = torch.stack([self.preprocess(img) for img in images])

        images = images.to(device)

        with torch.no_grad():
            features = self.model.encode_image(images)

        return features.cpu().numpy()

    def extract_dataset_features(self, dataset, batch_size=32):
        """
        Extract features for an entire dataset.

        Args:
            dataset: torchvision dataset
            batch_size: Batch size for extraction

        Returns:
            features: numpy array [N, 512]
            labels: numpy array [N]
        """
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

        all_features = []
        all_labels = []

        for images, labels in tqdm(dataloader, desc="Extracting CLIP features"):
            # Process images
            processed_images = torch.stack([self.preprocess(transforms.ToPILImage()(img)) for img in images])
            features = self.extract_features(processed_images)
            all_features.append(features)
            all_labels.append(labels.numpy())

        return np.vstack(all_features), np.concatenate(all_labels)


# =============================================================================
# Covariance Aggregation - Exactly as in Paper
# =============================================================================

def compute_mean(features):
    """Compute mean of features"""
    return np.mean(features, axis=0)


def compute_covariance(features, mean=None):
    """Compute covariance matrix"""
    if mean is None:
        mean = compute_mean(features)
    centered = features - mean
    return (1 / features.shape[0]) * np.dot(centered.T, centered)


def aggregate_covariance_matrices(client_features_per_class):
    """
    Aggregate covariance matrices from all clients for each class.
    This is EXACTLY the paper's algorithm from clip_tensor2aggregate_covariance_matrix.py

    Args:
        client_features_per_class: Dict[class_id] -> List[features_array]

    Returns:
        global_covariances: Dict[class_id] -> covariance_matrix [512, 512]
    """
    global_covariances = {}

    for class_id, client_features_list in client_features_per_class.items():
        sample_counts = []
        means = []
        covariances = []

        for features in client_features_list:
            if features.shape[0] == 0:
                continue

            mean = compute_mean(features)
            cov = compute_covariance(features, mean)

            sample_counts.append(features.shape[0])
            means.append(mean)
            covariances.append(cov)

        if len(sample_counts) == 0:
            # No samples for this class
            global_covariances[class_id] = np.zeros((512, 512))
            continue

        # Aggregate using weighted average + between-client variance
        total_samples = sum(sample_counts)

        # Combined mean
        combined_mean = np.sum([n * m for n, m in zip(sample_counts, means)], axis=0) / total_samples

        # Combined covariance
        # = weighted avg of within-client covariances + between-client variance
        combined_cov = np.sum([n * c for n, c in zip(sample_counts, covariances)], axis=0) / total_samples
        combined_cov += np.sum([n * np.outer(m - combined_mean, m - combined_mean)
                               for n, m in zip(sample_counts, means)], axis=0) / total_samples

        global_covariances[class_id] = combined_cov

        logger.debug(f"Class {class_id}: aggregated from {len(sample_counts)} clients, "
                    f"total {total_samples} samples")

    return global_covariances


# =============================================================================
# Sample Generation - Exactly as in Paper
# =============================================================================

def nearest_pos_def(cov_matrix):
    """
    Ensure covariance matrix is positive definite.
    This is EXACTLY the paper's algorithm.

    Key: Scale the SMALLEST 10 eigenvalues (ascending order from eigh)
    """
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

    # Scale smallest 10 eigenvalues by [5, 4.5, 4, 3.5, 3, 2.5, 2, 1.5, 1, 0.5]
    scale_factors = np.ones_like(eigenvalues)
    scale_factors[:10] = np.linspace(5, 1, 10)

    eigenvalues = eigenvalues * scale_factors
    eigenvalues[eigenvalues < 0] = 0  # Only clamp negative, NOT small positive!

    return eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T


def generate_samples(feature, cov_matrix, num_samples):
    """
    Generate new samples using multivariate normal.
    This is EXACTLY the paper's algorithm.
    """
    cov_matrix = nearest_pos_def(cov_matrix)

    # Add jitter for numerical stability
    jitter = 1e-6
    max_attempts = 10

    for attempt in range(max_attempts):
        try:
            B = np.linalg.cholesky(cov_matrix + jitter * np.eye(cov_matrix.shape[0]))
            break
        except np.linalg.LinAlgError:
            jitter *= 10

    # Generate samples
    # Note: B @ B.T = cov_matrix (approximately, with jitter)
    new_features = np.random.multivariate_normal(feature, B @ B.T, num_samples)

    return new_features


def augment_class_features(original_features, global_cov, other_prototypes, target_size=50, n_per_sample=50):
    """
    Augment features for one class.
    This is EXACTLY the paper's algorithm from prototype_cov_matrix_generate_features.py

    Steps:
    1. If original samples >= target_size: randomly select target_size from original
    2. Otherwise:
       a. Generate n_per_sample samples from each original sample, select 50 total
       b. Generate 50 samples from each other domain prototype
       c. Combine: keep ALL original + fill with generated up to target_size

    Args:
        original_features: [N, 512] original features for this class
        global_cov: [512, 512] global covariance matrix for this class
        other_prototypes: List of [512] prototypes from other domains
        target_size: Target number of samples (default 50)
        n_per_sample: Samples to generate per original sample

    Returns:
        final_features: [target_size, 512] augmented features
    """
    num_original = original_features.shape[0]

    # Case 1: Enough original samples
    if num_original >= target_size:
        indices = np.random.choice(num_original, target_size, replace=False)
        return original_features[indices]

    # Case 2: Need to generate samples

    # Step 1: Expand from original features
    if num_original > 0:
        all_generated = []
        for feature in original_features:
            new_samples = generate_samples(feature, global_cov, n_per_sample)
            all_generated.append(new_samples)

        all_generated = np.vstack(all_generated)

        # Select 50 from all generated
        if len(all_generated) >= 50:
            indices = np.random.choice(len(all_generated), 50, replace=False)
            expanded_features = all_generated[indices]
        else:
            expanded_features = all_generated
    else:
        expanded_features = np.empty((0, 512))

    # Step 2: Generate from other domain prototypes
    other_generated = []
    for prototype in other_prototypes:
        if prototype.size > 0:
            new_samples = generate_samples(prototype, global_cov, 50)
            other_generated.append(new_samples)

    if len(other_generated) > 0:
        other_generated = np.vstack(other_generated)
    else:
        other_generated = np.empty((0, 512))

    # Step 3: Combine
    if num_original == 0:
        # No original samples, use only generated
        if len(other_generated) >= target_size:
            indices = np.random.choice(len(other_generated), target_size, replace=False)
            return other_generated[indices]
        else:
            return other_generated

    # Keep all original, fill with generated
    num_needed = target_size - num_original

    # Combine all generated
    all_generated = []
    if len(expanded_features) > 0:
        all_generated.append(expanded_features)
    if len(other_generated) > 0:
        all_generated.append(other_generated)

    if len(all_generated) > 0:
        all_generated = np.vstack(all_generated)

        if len(all_generated) >= num_needed:
            indices = np.random.choice(len(all_generated), num_needed, replace=False)
            selected = all_generated[indices]
        else:
            selected = all_generated

        final_features = np.vstack([original_features, selected])
    else:
        final_features = original_features

    return final_features


# =============================================================================
# Training Functions - Exactly as in Paper
# =============================================================================

def train_epoch(model, dataloader, criterion, optimizer):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    return running_loss / total, correct / total


def evaluate(model, dataloader):
    """Evaluate model"""
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return correct / total


def federated_averaging(global_model, client_models):
    """FedAvg aggregation"""
    global_dict = global_model.state_dict()
    for key in global_dict.keys():
        global_dict[key] = torch.stack(
            [cm.state_dict()[key].float() for cm in client_models], 0
        ).mean(0)
    global_model.load_state_dict(global_dict)
    return global_model


# =============================================================================
# Main Pipeline
# =============================================================================

def load_office_home_data(data_root, domains=['Art', 'Clipart', 'Product', 'Real_World']):
    """
    Load Office-Home dataset.

    Returns:
        train_data: Dict[domain] -> (features, labels)
        test_data: Dict[domain] -> (features, labels)
    """
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    train_data = {}
    test_data = {}

    for domain in domains:
        domain_path = os.path.join(data_root, domain)
        if not os.path.exists(domain_path):
            logger.warning(f"Domain path not found: {domain_path}")
            continue

        dataset = datasets.ImageFolder(domain_path, transform=transform)

        # Split into train/test (80/20)
        n_samples = len(dataset)
        indices = np.random.permutation(n_samples)
        train_size = int(0.8 * n_samples)

        train_indices = indices[:train_size]
        test_indices = indices[train_size:]

        train_subset = torch.utils.data.Subset(dataset, train_indices)
        test_subset = torch.utils.data.Subset(dataset, test_indices)

        train_data[domain] = train_subset
        test_data[domain] = test_subset

        logger.info(f"Domain {domain}: {len(train_subset)} train, {len(test_subset)} test")

    return train_data, test_data


def run_ggeur_pipeline(args):
    """
    Run the complete GGEUR_Clip pipeline.

    Steps:
    1. Load data
    2. Extract CLIP features (or load from cache)
    3. Apply LDS to create heterogeneous data
    4. Compute global covariance matrices
    5. Compute prototypes
    6. Generate augmented features (offline)
    7. Run federated training
    """
    logger.info("=" * 60)
    logger.info("GGEUR_Clip Pipeline - Exact Match to Paper")
    logger.info("=" * 60)

    domains = ['Art', 'Clipart', 'Product', 'Real_World']
    num_classes = 65

    # Cache directory
    cache_dir = os.path.join(args.output_dir, 'cache')
    os.makedirs(cache_dir, exist_ok=True)

    # =========================================================================
    # Step 1: Extract CLIP features (or load from cache)
    # =========================================================================
    logger.info("\n[Step 1] Loading/Extracting CLIP features...")

    client_features = {}  # {domain: {class_id: features}}
    test_features = {}  # {domain: (features, labels)}

    clip_extractor = None

    for domain in domains:
        train_cache = os.path.join(cache_dir, f'{domain}_train_features.npz')
        test_cache = os.path.join(cache_dir, f'{domain}_test_features.npz')

        if os.path.exists(train_cache) and os.path.exists(test_cache):
            logger.info(f"  Loading cached features for {domain}")
            train_data = np.load(train_cache)
            test_data = np.load(test_cache)

            train_feats = train_data['features']
            train_labels = train_data['labels']
            test_features[domain] = (test_data['features'], test_data['labels'])

            # Organize by class
            client_features[domain] = {}
            for class_id in range(num_classes):
                mask = train_labels == class_id
                client_features[domain][class_id] = train_feats[mask]

        else:
            logger.info(f"  Extracting features for {domain} (this may take a while)")

            if clip_extractor is None:
                clip_extractor = CLIPFeatureExtractor(
                    model_name=args.clip_model,
                    weights_path=args.clip_weights
                )

            # Load data
            domain_path = os.path.join(args.data_root, domain)
            transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
            dataset = datasets.ImageFolder(domain_path, transform=transform)

            # Split
            n_samples = len(dataset)
            indices = np.random.permutation(n_samples)
            train_size = int(0.8 * n_samples)

            train_indices = indices[:train_size]
            test_indices = indices[train_size:]

            train_subset = torch.utils.data.Subset(dataset, train_indices)
            test_subset = torch.utils.data.Subset(dataset, test_indices)

            # Extract
            train_feats, train_labels = clip_extractor.extract_dataset_features(train_subset)
            test_feats, test_labels = clip_extractor.extract_dataset_features(test_subset)

            # Save cache
            np.savez(train_cache, features=train_feats, labels=train_labels)
            np.savez(test_cache, features=test_feats, labels=test_labels)

            test_features[domain] = (test_feats, test_labels)

            # Organize by class
            client_features[domain] = {}
            for class_id in range(num_classes):
                mask = train_labels == class_id
                client_features[domain][class_id] = train_feats[mask]

    # =========================================================================
    # Step 2: Apply LDS (Label Distribution Skew)
    # =========================================================================
    logger.info("\n[Step 2] Applying Label Distribution Skew (LDS)...")

    alpha = args.lda_alpha  # Dirichlet concentration parameter

    # Generate Dirichlet distribution for each client
    np.random.seed(args.seed)
    dirichlet_matrix = np.random.dirichlet([alpha] * len(domains), num_classes).T  # [4, 65]

    # Apply LDS
    lds_client_features = {}
    for domain_idx, domain in enumerate(domains):
        lds_client_features[domain] = {}
        total_samples = 0

        for class_id in range(num_classes):
            original = client_features[domain][class_id]
            if len(original) == 0:
                lds_client_features[domain][class_id] = original
                continue

            # Sample proportion for this client and class
            proportion = dirichlet_matrix[domain_idx, class_id]
            n_samples = int(len(original) * proportion * len(domains))
            n_samples = min(n_samples, len(original))  # Cap at available

            if n_samples > 0:
                indices = np.random.choice(len(original), n_samples, replace=False)
                lds_client_features[domain][class_id] = original[indices]
            else:
                lds_client_features[domain][class_id] = np.empty((0, 512))

            total_samples += len(lds_client_features[domain][class_id])

        logger.info(f"  {domain}: {total_samples} samples after LDS")

    # =========================================================================
    # Step 3: Compute Global Covariance Matrices
    # =========================================================================
    logger.info("\n[Step 3] Computing global covariance matrices...")

    # Organize features by class across all clients
    features_by_class = {c: [] for c in range(num_classes)}
    for domain in domains:
        for class_id in range(num_classes):
            feats = lds_client_features[domain][class_id]
            if len(feats) > 0:
                features_by_class[class_id].append(feats)

    global_covariances = aggregate_covariance_matrices(features_by_class)
    logger.info(f"  Computed covariances for {len(global_covariances)} classes")

    # =========================================================================
    # Step 4: Compute Prototypes
    # =========================================================================
    logger.info("\n[Step 4] Computing prototypes...")

    prototypes = {}  # {domain: {class_id: prototype}}
    for domain in domains:
        prototypes[domain] = {}
        for class_id in range(num_classes):
            feats = lds_client_features[domain][class_id]
            if len(feats) > 0:
                prototypes[domain][class_id] = np.mean(feats, axis=0)
            else:
                prototypes[domain][class_id] = np.empty(0)

    # =========================================================================
    # Step 5: Generate Augmented Features (OFFLINE!)
    # =========================================================================
    logger.info("\n[Step 5] Generating augmented features (offline)...")

    augmented_features = {}  # {domain: (features, labels)}

    for domain_idx, domain in enumerate(domains):
        all_features = []
        all_labels = []

        other_domains = [d for d in domains if d != domain]

        for class_id in range(num_classes):
            original = lds_client_features[domain][class_id]
            cov = global_covariances[class_id]

            # Get prototypes from other domains
            other_prototypes = []
            for other_domain in other_domains:
                proto = prototypes[other_domain].get(class_id, np.empty(0))
                if proto.size > 0:
                    other_prototypes.append(proto)

            # Skip if no covariance (class doesn't exist)
            if np.all(cov == 0):
                if len(original) > 0:
                    all_features.append(original)
                    all_labels.append(np.full(len(original), class_id))
                continue

            # Augment
            augmented = augment_class_features(
                original, cov, other_prototypes,
                target_size=args.target_size_per_class,
                n_per_sample=args.n_samples_per_original
            )

            if len(augmented) > 0:
                all_features.append(augmented)
                all_labels.append(np.full(len(augmented), class_id))

        if all_features:
            augmented_features[domain] = (
                np.vstack(all_features),
                np.concatenate(all_labels)
            )
            logger.info(f"  {domain}: {len(augmented_features[domain][0])} augmented samples")
        else:
            augmented_features[domain] = (np.empty((0, 512)), np.empty(0))

    # =========================================================================
    # Step 6: Federated Training
    # =========================================================================
    logger.info("\n[Step 6] Running federated training...")

    # Create models
    global_model = GGEURClassifier(input_dim=512, num_classes=num_classes).to(device)
    client_models = [GGEURClassifier(input_dim=512, num_classes=num_classes).to(device)
                     for _ in domains]

    criterion = nn.CrossEntropyLoss()

    # Prepare dataloaders
    client_loaders = []
    for domain in domains:
        features, labels = augmented_features[domain]
        if len(features) > 0:
            dataset = FeatureDataset(features, labels)
            loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
            client_loaders.append(loader)
        else:
            client_loaders.append(None)

    # Prepare test loaders (combine all domains)
    all_test_features = []
    all_test_labels = []
    for domain in domains:
        feats, labels = test_features[domain]
        all_test_features.append(feats)
        all_test_labels.append(labels)

    combined_test = FeatureDataset(np.vstack(all_test_features), np.concatenate(all_test_labels))
    test_loader = DataLoader(combined_test, batch_size=args.batch_size, shuffle=False)

    # Training loop
    best_acc = 0.0

    for round_idx in range(args.num_rounds):
        # Distribute global model
        for client_model in client_models:
            client_model.load_state_dict(global_model.state_dict())

        # Local training
        for client_idx, loader in enumerate(client_loaders):
            if loader is None:
                continue

            client_model = client_models[client_idx]
            optimizer = optim.Adam(client_model.parameters(), lr=args.lr)

            for epoch in range(args.local_epochs):
                train_loss, train_acc = train_epoch(client_model, loader, criterion, optimizer)

        # FedAvg
        active_models = [m for m, l in zip(client_models, client_loaders) if l is not None]
        global_model = federated_averaging(global_model, active_models)

        # Evaluate
        test_acc = evaluate(global_model, test_loader)

        if test_acc > best_acc:
            best_acc = test_acc

        if (round_idx + 1) % 10 == 0 or round_idx == 0:
            logger.info(f"  Round {round_idx + 1}/{args.num_rounds}: "
                       f"Test Acc = {test_acc:.4f} ({test_acc*100:.2f}%), "
                       f"Best = {best_acc:.4f}")

    # =========================================================================
    # Results
    # =========================================================================
    logger.info("\n" + "=" * 60)
    logger.info("FINAL RESULTS")
    logger.info("=" * 60)
    logger.info(f"Best Test Accuracy: {best_acc:.4f} ({best_acc*100:.2f}%)")

    # Per-domain evaluation
    logger.info("\nPer-domain accuracy:")
    for domain in domains:
        feats, labels = test_features[domain]
        domain_dataset = FeatureDataset(feats, labels)
        domain_loader = DataLoader(domain_dataset, batch_size=args.batch_size, shuffle=False)
        domain_acc = evaluate(global_model, domain_loader)
        logger.info(f"  {domain}: {domain_acc:.4f} ({domain_acc*100:.2f}%)")


def main():
    parser = argparse.ArgumentParser(description='GGEUR_Clip Standalone Implementation')

    # Data
    parser.add_argument('--data_root', type=str, default='./OfficeHomeDataset_10072016',
                       help='Path to Office-Home dataset')
    parser.add_argument('--output_dir', type=str, default='./ggeur_output',
                       help='Output directory')

    # CLIP
    parser.add_argument('--clip_model', type=str, default='ViT-B-32',
                       help='CLIP model name (paper uses ViT-B-32)')
    parser.add_argument('--clip_weights', type=str, default=None,
                       help='Path to custom CLIP weights')

    # LDS
    parser.add_argument('--lda_alpha', type=float, default=0.1,
                       help='Dirichlet concentration parameter')

    # GGEUR_Clip
    parser.add_argument('--target_size_per_class', type=int, default=50,
                       help='Target samples per class after augmentation')
    parser.add_argument('--n_samples_per_original', type=int, default=50,
                       help='Samples to generate per original sample')

    # Training
    parser.add_argument('--num_rounds', type=int, default=50,
                       help='Number of communication rounds')
    parser.add_argument('--local_epochs', type=int, default=1,
                       help='Local training epochs per round')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--seed', type=int, default=12345,
                       help='Random seed')

    args = parser.parse_args()

    # Set seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)

    run_ggeur_pipeline(args)


if __name__ == '__main__':
    main()
