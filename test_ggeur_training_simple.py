"""
Quick test: Check if training data setup is the problem
"""
import sys
sys.path.append('D:/Projects/FederatedScope')

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import TensorDataset, DataLoader

print("="*80)
print("Test: Load and check actual GGEUR_Clip augmented data format")
print("="*80)

# Simulate what GGEUR_Clip does
np.random.seed(42)
torch.manual_seed(42)

# Create fake augmented embeddings (normalized, like CLIP features)
n_samples = 3250  # 65 classes * 50 samples
aug_embeddings = torch.randn(n_samples, 512)
aug_embeddings = aug_embeddings / aug_embeddings.norm(dim=-1, keepdim=True)  # CRITICAL: L2 normalize
aug_labels = torch.arange(65).repeat_interleave(50)

print(f"Augmented data:")
print(f"  Shape: {aug_embeddings.shape}")
print(f"  Labels: {aug_labels.shape}, unique: {torch.unique(aug_labels).tolist()[:10]}...")
print(f"  Norm mean: {aug_embeddings.norm(dim=-1).mean():.6f}")
print(f"  Norm std: {aug_embeddings.norm(dim=-1).std():.6f}")
print(f"  Value range: [{aug_embeddings.min():.4f}, {aug_embeddings.max():.4f}]")

# Create model WITH softmax (like original)
class SoftmaxMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(512, 65)

    def forward(self, x):
        logits = self.fc(x)
        return torch.softmax(logits, dim=1)

model = SoftmaxMLP()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Create dataloader
dataset = TensorDataset(aug_embeddings, aug_labels)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

print(f"\nDataLoader:")
print(f"  Batches: {len(dataloader)}")
print(f"  Batch size: 16")

# Train for 1 epoch (like original local_epochs=1)
print(f"\nTraining for 1 epoch...")
model.train()

total_loss = 0
total_correct = 0
total_samples = 0
batch_losses = []

for batch_idx, (x, y) in enumerate(dataloader):
    optimizer.zero_grad()
    outputs = model(x)
    loss = criterion(outputs, y)

    # Check if loss is valid
    if torch.isnan(loss) or torch.isinf(loss):
        print(f"  WARNING: Invalid loss at batch {batch_idx}: {loss.item()}")
        break

    loss.backward()

    # Check gradients
    if batch_idx == 0:
        grad_norm = model.fc.weight.grad.norm().item()
        print(f"  First batch gradient norm: {grad_norm:.6f}")
        if grad_norm < 1e-6:
            print(f"    WARNING: Gradient is too small!")

    optimizer.step()

    total_loss += loss.item() * len(x)
    pred = outputs.argmax(dim=1)
    total_correct += (pred == y).sum().item()
    total_samples += len(x)
    batch_losses.append(loss.item())

avg_loss = total_loss / total_samples
avg_acc = total_correct / total_samples

print(f"\nAfter 1 epoch:")
print(f"  Avg Loss: {avg_loss:.4f}")
print(f"  Avg Acc: {avg_acc:.4f} ({avg_acc*100:.2f}%)")
print(f"  Loss std: {np.std(batch_losses):.4f}")
print(f"  First 5 batch losses: {batch_losses[:5]}")
print(f"  Last 5 batch losses: {batch_losses[-5:]}")

if avg_acc < 0.05:
    print("\n  PROBLEM FOUND: Model cannot learn!")
    print("  This matches the actual bug.")
    print("\n  Possible causes:")
    print("  1. Data is corrupted/all zeros")
    print("  2. Labels are wrong")
    print("  3. Softmax + CrossEntropyLoss issue")
    print("  4. Learning rate too small")
else:
    print("\n  Model CAN learn in this test.")
    print("  Problem must be elsewhere in GGEUR_Clip implementation.")

# Additional check: are predictions all the same?
model.eval()
with torch.no_grad():
    test_output = model(aug_embeddings[:100])
    test_pred = test_output.argmax(dim=1)
    pred_counts = torch.bincount(test_pred, minlength=65)
    max_pred_class = pred_counts.argmax().item()
    max_pred_count = pred_counts.max().item()

    print(f"\nPrediction distribution (first 100 samples):")
    print(f"  Most predicted class: {max_pred_class} ({max_pred_count}/100 samples)")
    print(f"  Unique predictions: {len(torch.unique(test_pred))}/65 classes")

    if max_pred_count > 80:
        print(f"    PROBLEM: Model is heavily biased to one class!")

print("="*80)
