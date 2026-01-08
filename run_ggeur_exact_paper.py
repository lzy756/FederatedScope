"""
Exact replication of paper's GGEUR_Clip implementation for debugging.
This script replicates FedAvg_GGEUR.py from the paper exactly.
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Custom Dataset
class MyDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features if isinstance(features, torch.Tensor) else torch.from_numpy(features).float()
        self.labels = labels if isinstance(labels, torch.Tensor) else torch.from_numpy(labels).long()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


# Model - EXACTLY as in paper
class MyNet(nn.Module):
    def __init__(self, num_classes=65):
        super(MyNet, self).__init__()
        self.fc3 = nn.Linear(512, num_classes)

    def forward(self, x):
        # Paper uses softmax in forward!
        return F.softmax(self.fc3(x), dim=1)


# Alternative model WITHOUT softmax (for comparison)
class MyNetLogits(nn.Module):
    def __init__(self, num_classes=65):
        super(MyNetLogits, self).__init__()
        self.fc3 = nn.Linear(512, num_classes)

    def forward(self, x):
        return self.fc3(x)  # Return logits


# Train function - EXACTLY as in paper
def train_paper_style(model, dataloader, criterion, optimizer, device):
    """Paper's training with softmax output + CrossEntropyLoss"""
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)  # outputs are probabilities (after softmax)
        loss = criterion(outputs, labels)  # CrossEntropyLoss on probabilities!
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_accuracy = correct / total if total > 0 else 0
    return epoch_loss, epoch_accuracy


# Train function - CORRECT version
def train_correct(model, dataloader, criterion, optimizer, device):
    """Correct training with logits output + CrossEntropyLoss"""
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)  # outputs are logits
        loss = criterion(outputs, labels)  # CrossEntropyLoss on logits (correct!)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_accuracy = correct / total if total > 0 else 0
    return epoch_loss, epoch_accuracy


# Test function
def test(model, dataloader, device):
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


# FedAvg
def federated_averaging(global_model, client_models):
    global_dict = global_model.state_dict()
    for k in global_dict.keys():
        global_dict[k] = torch.stack([client_models[i].state_dict()[k].float() for i in range(len(client_models))], 0).mean(0)
    global_model.load_state_dict(global_dict)
    return global_model


def generate_synthetic_data(num_clients=4, num_classes=65, samples_per_class=50, feature_dim=512):
    """Generate synthetic CLIP-like features for testing"""
    print("Generating synthetic data...")

    all_client_features = []

    # Create class centers (simulating CLIP embeddings)
    class_centers = np.random.randn(num_classes, feature_dim).astype(np.float32)
    class_centers = class_centers / np.linalg.norm(class_centers, axis=1, keepdims=True)

    for client_id in range(num_clients):
        features_list = []
        labels_list = []

        # Each client has different classes (simulating heterogeneity)
        client_classes = np.random.choice(num_classes, size=int(num_classes * 0.5), replace=False)

        for class_id in client_classes:
            # Generate samples around class center with some noise
            class_features = class_centers[class_id] + 0.1 * np.random.randn(samples_per_class, feature_dim).astype(np.float32)
            # Normalize to match CLIP
            class_features = class_features / np.linalg.norm(class_features, axis=1, keepdims=True)

            features_list.append(class_features)
            labels_list.append(np.full(samples_per_class, class_id, dtype=np.int64))

        features = np.vstack(features_list)
        labels = np.concatenate(labels_list)

        all_client_features.append((features, labels))
        print(f"  Client {client_id}: {len(features)} samples, {len(client_classes)} classes")

    # Generate test set (all classes)
    test_features_list = []
    test_labels_list = []
    test_samples_per_class = 10

    for class_id in range(num_classes):
        class_features = class_centers[class_id] + 0.1 * np.random.randn(test_samples_per_class, feature_dim).astype(np.float32)
        class_features = class_features / np.linalg.norm(class_features, axis=1, keepdims=True)
        test_features_list.append(class_features)
        test_labels_list.append(np.full(test_samples_per_class, class_id, dtype=np.int64))

    test_features = np.vstack(test_features_list)
    test_labels = np.concatenate(test_labels_list)
    print(f"  Test set: {len(test_features)} samples, {num_classes} classes")

    return all_client_features, (test_features, test_labels)


def run_experiment(use_paper_style=True, num_rounds=50, local_epochs=1, lr=0.001):
    """
    Run federated learning experiment.

    Args:
        use_paper_style: If True, use paper's softmax+CrossEntropyLoss
                        If False, use correct logits+CrossEntropyLoss
    """
    print(f"\n{'='*60}")
    print(f"Experiment: {'Paper Style (softmax+CE)' if use_paper_style else 'Correct Style (logits+CE)'}")
    print(f"Rounds: {num_rounds}, Local Epochs: {local_epochs}, LR: {lr}")
    print(f"{'='*60}")

    # Generate data
    all_client_features, test_data = generate_synthetic_data()
    test_features, test_labels = test_data
    test_dataset = MyDataset(test_features, test_labels)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    num_clients = len(all_client_features)

    # Create models
    if use_paper_style:
        global_model = MyNet(num_classes=65).to(device)
        client_models = [MyNet(num_classes=65).to(device) for _ in range(num_clients)]
        train_fn = train_paper_style
    else:
        global_model = MyNetLogits(num_classes=65).to(device)
        client_models = [MyNetLogits(num_classes=65).to(device) for _ in range(num_clients)]
        train_fn = train_correct

    criterion = nn.CrossEntropyLoss()

    # Training loop
    for round_idx in range(num_rounds):
        # Distribute global model to clients
        for client_model in client_models:
            client_model.load_state_dict(global_model.state_dict())

        # Local training
        for client_idx, client_data in enumerate(all_client_features):
            features, labels = client_data
            client_model = client_models[client_idx]
            optimizer = optim.Adam(client_model.parameters(), lr=lr)

            train_dataset = MyDataset(features, labels)
            train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

            for epoch in range(local_epochs):
                train_loss, train_acc = train_fn(client_model, train_loader, criterion, optimizer, device)

        # FedAvg aggregation
        global_model = federated_averaging(global_model, client_models)

        # Evaluate
        test_acc = test(global_model, test_loader, device)

        if (round_idx + 1) % 10 == 0 or round_idx == 0:
            print(f"Round {round_idx + 1}/{num_rounds}: Test Accuracy = {test_acc:.4f} ({test_acc*100:.2f}%)")

    final_acc = test(global_model, test_loader, device)
    print(f"\nFinal Test Accuracy: {final_acc:.4f} ({final_acc*100:.2f}%)")

    return final_acc


if __name__ == "__main__":
    print("Testing paper's implementation style vs correct style")
    print("This will help identify if the softmax+CrossEntropyLoss issue is the root cause")

    # Run paper style
    paper_acc = run_experiment(use_paper_style=True, num_rounds=50)

    # Run correct style
    correct_acc = run_experiment(use_paper_style=False, num_rounds=50)

    print(f"\n{'='*60}")
    print("COMPARISON RESULTS")
    print(f"{'='*60}")
    print(f"Paper Style (softmax + CrossEntropyLoss): {paper_acc:.4f} ({paper_acc*100:.2f}%)")
    print(f"Correct Style (logits + CrossEntropyLoss): {correct_acc:.4f} ({correct_acc*100:.2f}%)")
    print(f"Difference: {(correct_acc - paper_acc)*100:.2f}%")

    if correct_acc > paper_acc + 0.1:
        print("\n>>> CONCLUSION: Softmax + CrossEntropyLoss is likely the root cause!")
    elif abs(correct_acc - paper_acc) < 0.1:
        print("\n>>> CONCLUSION: Both methods perform similarly. Issue may be elsewhere.")
    else:
        print("\n>>> CONCLUSION: Paper style performs better? Unexpected result.")
