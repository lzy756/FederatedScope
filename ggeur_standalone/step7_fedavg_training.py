"""
GGEUR_Clip Step 7: FedAvg Training

This script:
1. Loads augmented CLIP features for each client
2. Performs Federated Averaging training
3. Evaluates on test sets from all domains
4. Saves best model and training logs
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import matplotlib.pyplot as plt
import argparse


# Client ID assignment
CLIENT_RANGE = {
    'Art': [0],
    'Clipart': [1],
    'Product': [2],
    'Real_World': [3]
}


class MyDataset(Dataset):
    """Simple dataset for CLIP features"""
    def __init__(self, features, labels):
        self.features = features if isinstance(features, torch.Tensor) else torch.from_numpy(features).float()
        self.labels = labels if isinstance(labels, torch.Tensor) else torch.from_numpy(labels).long()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


class MyNet(nn.Module):
    """Simple linear classifier for CLIP features"""
    def __init__(self, num_classes=65, feature_dim=512):
        super(MyNet, self).__init__()
        self.fc = nn.Linear(feature_dim, num_classes)

    def forward(self, x):
        return F.softmax(self.fc(x), dim=1)


def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    running_loss, correct, total = 0.0, 0, 0

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

    epoch_loss = running_loss / len(dataloader.dataset) if len(dataloader.dataset) > 0 else 0
    epoch_accuracy = correct / total if total > 0 else 0
    return epoch_loss, epoch_accuracy


def test(model, dataloader, device):
    """Evaluate model on test set"""
    model.eval()
    correct, total = 0, 0

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total if total > 0 else 0
    return accuracy


def federated_averaging(global_model, client_models):
    """Aggregate client models using FedAvg"""
    global_dict = global_model.state_dict()

    for k in global_dict.keys():
        global_dict[k] = torch.stack([
            client_models[i].state_dict()[k].float()
            for i in range(len(client_models))
        ], 0).mean(0)

    global_model.load_state_dict(global_dict)
    return global_model


def load_client_features(client_idx, dataset_name, base_dir, num_classes=65):
    """Load augmented features for a client"""
    features, labels = [], []

    for class_idx in range(num_classes):
        features_path = os.path.join(base_dir, dataset_name, f'client_{client_idx}_class_{class_idx}', 'final_embeddings_filled.npy')
        labels_path = os.path.join(base_dir, dataset_name, f'client_{client_idx}_class_{class_idx}', 'labels_filled.npy')

        if os.path.exists(features_path) and os.path.exists(labels_path):
            class_features = np.load(features_path)
            class_labels = np.load(labels_path)
            features.append(class_features)
            labels.append(class_labels)

    if features and labels:
        features = np.vstack(features)
        labels = np.concatenate(labels)
        return features, labels
    else:
        return None, None


def load_test_features_labels(dataset_name, base_dir):
    """Load test features for a domain"""
    test_features_path = os.path.join(base_dir, dataset_name, f'{dataset_name}_test_features.npy')
    test_labels_path = os.path.join(base_dir, dataset_name, f'{dataset_name}_test_labels.npy')

    if os.path.exists(test_features_path) and os.path.exists(test_labels_path):
        test_features = np.load(test_features_path)
        test_labels = np.load(test_labels_path)
        return torch.tensor(test_features, dtype=torch.float32), torch.tensor(test_labels, dtype=torch.long)
    else:
        raise FileNotFoundError(f"Test features for {dataset_name} not found")


def plot_accuracies(accuracies, communication_rounds, output_path):
    """Plot accuracy curves"""
    plt.figure(figsize=(10, 6))
    rounds = range(1, communication_rounds + 1)

    for dataset_name, acc_values in accuracies.items():
        plt.plot(rounds, acc_values, label=f'{dataset_name}')

    plt.xlabel('Communication Rounds')
    plt.ylabel('Accuracy')
    plt.title('Test Accuracy across Communication Rounds')
    plt.legend()
    plt.grid(True)
    plt.savefig(output_path)
    plt.close()
    print(f"Saved accuracy plot to {output_path}")


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("="*60)
    print("FedAvg Training with GGEUR_Clip Augmented Features")
    print("="*60)

    # Paths
    augmented_dir = os.path.join(args.workspace, 'augmented_features')
    test_dir = os.path.join(args.workspace, 'clip_test_features')
    results_dir = os.path.join(args.workspace, 'results')
    model_dir = os.path.join(args.workspace, 'model')
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    datasets = ['Art', 'Clipart', 'Product', 'Real_World']

    # Load client features
    print("\nLoading client features...")
    all_client_features = []
    for dataset_name in datasets:
        for client_id in CLIENT_RANGE[dataset_name]:
            features, labels = load_client_features(client_id, dataset_name, augmented_dir)
            if features is not None and labels is not None:
                all_client_features.append((features, labels))
                print(f"  {dataset_name} client {client_id}: {len(labels)} samples")

    if len(all_client_features) == 0:
        print("Error: No client features loaded!")
        return

    # Load test sets
    print("\nLoading test sets...")
    test_sets = {}
    for dataset_name in datasets:
        try:
            test_features, test_labels = load_test_features_labels(dataset_name, test_dir)
            test_sets[dataset_name] = (test_features, test_labels)
            print(f"  {dataset_name}: {len(test_labels)} test samples")
        except FileNotFoundError as e:
            print(f"Warning: {e}")

    if len(test_sets) == 0:
        print("Error: No test sets loaded!")
        return

    # Initialize models
    global_model = MyNet(num_classes=65).to(device)
    client_models = [MyNet(num_classes=65).to(device) for _ in range(len(all_client_features))]

    criterion = nn.CrossEntropyLoss()

    # Training logs
    all_accuracies = {name: [] for name in list(test_sets.keys()) + ['average']}
    best_avg_accuracy = 0.0
    best_model_state = None

    report_path = os.path.join(results_dir, 'training_log.txt')

    print(f"\nStarting training for {args.communication_rounds} rounds...")
    print("="*60)

    with open(report_path, 'w') as f:
        f.write("GGEUR_Clip FedAvg Training Log\n")
        f.write("="*60 + "\n\n")

        for round_idx in range(args.communication_rounds):
            # Local training
            for client_idx, (features, labels) in enumerate(all_client_features):
                client_model = client_models[client_idx]

                # Copy global model to client
                client_model.load_state_dict(global_model.state_dict())

                optimizer = optim.Adam(client_model.parameters(), lr=args.learning_rate)

                for epoch in range(args.local_epochs):
                    train_dataset = MyDataset(features, labels)
                    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
                    train_epoch(client_model, train_dataloader, criterion, optimizer, device)

            # FedAvg aggregation
            global_model = federated_averaging(global_model, client_models)

            # Evaluate on all test sets
            accuracies = []
            for dataset_name, (test_features, test_labels) in test_sets.items():
                test_dataset = MyDataset(test_features, test_labels)
                test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
                test_accuracy = test(global_model, test_dataloader, device)
                accuracies.append(test_accuracy)
                all_accuracies[dataset_name].append(test_accuracy)

            avg_accuracy = sum(accuracies) / len(accuracies)
            all_accuracies['average'].append(avg_accuracy)

            # Log
            accuracy_str = ', '.join([f"{name}: {acc:.4f}" for name, acc in zip(test_sets.keys(), accuracies)])
            log_line = f"Round {round_idx + 1}/{args.communication_rounds}, {accuracy_str}, Average: {avg_accuracy:.4f}"
            print(log_line)
            f.write(log_line + "\n")

            # Save best model
            if avg_accuracy > best_avg_accuracy:
                best_avg_accuracy = avg_accuracy
                best_model_state = global_model.state_dict()

    # Save best model
    best_model_path = os.path.join(model_dir, 'best_model.pth')
    torch.save(best_model_state, best_model_path)
    print(f"\nBest model saved to {best_model_path}")
    print(f"Best average accuracy: {best_avg_accuracy:.4f}")

    # Plot accuracies
    plot_path = os.path.join(results_dir, 'accuracy_plot.png')
    plot_accuracies(all_accuracies, args.communication_rounds, plot_path)

    print("\n" + "="*60)
    print("Training completed!")
    print("="*60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='GGEUR_Clip Step 7: FedAvg Training')
    parser.add_argument('--workspace', type=str,
                        default='./ggeur_standalone/workspace',
                        help='Workspace directory')
    parser.add_argument('--communication_rounds', type=int, default=50,
                        help='Number of communication rounds')
    parser.add_argument('--local_epochs', type=int, default=1,
                        help='Local training epochs per round')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Learning rate')
    args = parser.parse_args()

    main(args)
