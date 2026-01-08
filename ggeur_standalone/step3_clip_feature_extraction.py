"""
GGEUR_Clip Step 3: CLIP Feature Extraction

This script:
1. Extracts CLIP features for training samples (per-client per-class)
2. Extracts CLIP features for test samples

Uses OpenCLIP with ViT-B-32 backbone
"""

import os
import numpy as np
import torch
from PIL import Image
import open_clip
import argparse
from tqdm import tqdm
from torchvision import datasets, transforms


# Client ID assignment
CLIENT_RANGE = {
    'Art': [0],
    'Clipart': [1],
    'Product': [2],
    'Real_World': [3]
}


def get_domain_folder_name(domain, data_path):
    """Get actual folder name for domain (handles 'Real World' vs 'Real_World')"""
    if os.path.exists(os.path.join(data_path, domain)):
        return domain
    domain_with_space = domain.replace('_', ' ')
    if os.path.exists(os.path.join(data_path, domain_with_space)):
        return domain_with_space
    domain_with_underscore = domain.replace(' ', '_')
    if os.path.exists(os.path.join(data_path, domain_with_underscore)):
        return domain_with_underscore
    return domain


def create_clip_model(pretrained_path, backbone='ViT-B-16'):
    """Create CLIP model and preprocessing function"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if pretrained_path and os.path.exists(pretrained_path):
        print(f"Loading CLIP model from {pretrained_path}")
        print(f"Backbone: {backbone}")
        model, _, preprocess = open_clip.create_model_and_transforms(backbone, pretrained=pretrained_path)
    else:
        print(f"Loading CLIP model from OpenAI pretrained weights")
        print(f"Backbone: {backbone}")
        model, _, preprocess = open_clip.create_model_and_transforms(backbone, pretrained='openai')

    model.eval().to(device)
    return model, preprocess, device


def clip_image_embedding(image, model, preprocess, device):
    """Extract CLIP image features"""
    image = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        image_features = model.encode_image(image)
    return image_features.cpu().numpy().squeeze(0)


# Transform for Office-Home
transform_office_home = transforms.Compose([
    transforms.Resize((224, 224)),
])


def load_office_home_with_indices(path, domain, indices):
    """Load Office-Home subset with given indices"""
    # Get actual folder name
    folder_name = get_domain_folder_name(domain, path)
    domain_path = os.path.join(path, folder_name)
    dataset = datasets.ImageFolder(domain_path, transform=transform_office_home)
    subset = torch.utils.data.Subset(dataset, indices)
    return subset


def load_train_indices(client_id, class_label, domain, base_dir):
    """Load training indices for a specific client and class"""
    indices_path = os.path.join(base_dir, domain, f'client_{client_id}_class_{class_label}_indices.npy')
    if os.path.exists(indices_path):
        return np.load(indices_path)
    return np.array([])


def load_test_indices(domain, base_dir):
    """Load test indices for a domain"""
    indices_path = os.path.join(base_dir, domain, 'test_test_indices.npy')
    if os.path.exists(indices_path):
        return np.load(indices_path)
    return np.array([])


def process_train_set(dataset_name, client_id, class_label, data, output_dir, model, preprocess, device):
    """Process training set and extract features"""
    if len(data) == 0:
        print(f"  Skipping {dataset_name} client {client_id} class {class_label} - no samples")
        # Save empty arrays
        np.save(os.path.join(output_dir, f'client_{client_id}_class_{class_label}_original_features.npy'), np.array([]))
        np.save(os.path.join(output_dir, f'client_{client_id}_class_{class_label}_labels.npy'), np.array([]))
        return

    all_features = []
    labels = []

    for img, label in data:
        feature = clip_image_embedding(img, model, preprocess, device)
        all_features.append(feature)
        labels.append(label)

    all_features = np.array(all_features)
    labels = np.array(labels)

    np.save(os.path.join(output_dir, f'client_{client_id}_class_{class_label}_original_features.npy'), all_features)
    np.save(os.path.join(output_dir, f'client_{client_id}_class_{class_label}_labels.npy'), labels)

    print(f"  {dataset_name} client {client_id} class {class_label}: {len(all_features)} features")


def process_test_set(dataset_name, data, output_dir, model, preprocess, device):
    """Process test set and extract features"""
    all_features = []
    labels = []

    for img, label in tqdm(data, desc=f"Extracting {dataset_name} test features"):
        feature = clip_image_embedding(img, model, preprocess, device)
        all_features.append(feature)
        labels.append(label)

    all_features = np.array(all_features)
    labels = np.array(labels)

    np.save(os.path.join(output_dir, f'{dataset_name}_test_features.npy'), all_features)
    np.save(os.path.join(output_dir, f'{dataset_name}_test_labels.npy'), labels)

    print(f"Saved {dataset_name} test features: {len(all_features)} samples")


def extract_train_features(args, model, preprocess, device):
    """Extract training features for all clients and classes"""
    print("\n" + "="*60)
    print("Extracting TRAINING features")
    print("="*60)

    indices_dir = os.path.join(args.workspace, 'output_client_class_indices')
    output_dir = os.path.join(args.workspace, 'clip_train_features')
    os.makedirs(output_dir, exist_ok=True)

    domains = ['Art', 'Clipart', 'Product', 'Real_World']

    for dataset_name in domains:
        print(f"\nProcessing {dataset_name}...")
        domain_output_dir = os.path.join(output_dir, dataset_name)
        os.makedirs(domain_output_dir, exist_ok=True)

        assigned_clients = CLIENT_RANGE[dataset_name]

        for client_id in assigned_clients:
            for class_label in range(65):
                train_indices = load_train_indices(client_id, class_label, dataset_name, indices_dir)

                if len(train_indices) == 0:
                    # Save empty arrays
                    np.save(os.path.join(domain_output_dir, f'client_{client_id}_class_{class_label}_original_features.npy'), np.array([]))
                    np.save(os.path.join(domain_output_dir, f'client_{client_id}_class_{class_label}_labels.npy'), np.array([]))
                    continue

                data = load_office_home_with_indices(args.data_path, dataset_name, train_indices)
                process_train_set(dataset_name, client_id, class_label, data, domain_output_dir, model, preprocess, device)


def extract_test_features(args, model, preprocess, device):
    """Extract test features for all domains"""
    print("\n" + "="*60)
    print("Extracting TEST features")
    print("="*60)

    indices_dir = os.path.join(args.workspace, 'output_indices')
    output_dir = os.path.join(args.workspace, 'clip_test_features')
    os.makedirs(output_dir, exist_ok=True)

    domains = ['Art', 'Clipart', 'Product', 'Real_World']

    for dataset_name in domains:
        domain_output_dir = os.path.join(output_dir, dataset_name)
        os.makedirs(domain_output_dir, exist_ok=True)

        test_indices = load_test_indices(dataset_name, indices_dir)

        if len(test_indices) == 0:
            print(f"Warning: No test indices for {dataset_name}")
            continue

        data = load_office_home_with_indices(args.data_path, dataset_name, test_indices)
        process_test_set(dataset_name, data, domain_output_dir, model, preprocess, device)


def main(args):
    print("Creating CLIP model...")
    model, preprocess, device = create_clip_model(args.clip_model_path, args.backbone)
    print(f"Using device: {device}")

    if args.extract_train:
        extract_train_features(args, model, preprocess, device)

    if args.extract_test:
        extract_test_features(args, model, preprocess, device)

    print("\n" + "="*60)
    print("Feature extraction completed!")
    print("="*60)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GGEUR_Clip Step 3: CLIP Feature Extraction')
    parser.add_argument('--data_path', type=str,
                        default='/root/CDA_new/OfficeHomeDataset_10072016',
                        help='Path to Office-Home dataset')
    parser.add_argument('--workspace', type=str,
                        default='./ggeur_standalone/workspace',
                        help='Workspace directory')
    parser.add_argument('--clip_model_path', type=str,
                        default='/root/model/open_clip_vitb16.bin',
                        help='Path to CLIP model weights')
    parser.add_argument('--backbone', type=str,
                        default='ViT-B-16',
                        help='CLIP backbone architecture')
    parser.add_argument('--extract_train', action='store_true', default=True,
                        help='Extract training features')
    parser.add_argument('--extract_test', action='store_true', default=True,
                        help='Extract test features')
    args = parser.parse_args()

    main(args)
