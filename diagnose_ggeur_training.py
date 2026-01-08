"""
Diagnostic script to identify why GGEUR_Clip training fails
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import TensorDataset, DataLoader

print("="*80)
print("GGEUR_Clip Training Diagnostic")
print("="*80)

# ===== Test 1: Verify Softmax + CrossEntropyLoss can learn =====
print("\n" + "="*80)
print("Test 1: Can Softmax + CrossEntropyLoss learn?")
print("="*80)

class SoftmaxNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(512, 65)

    def forward(self, x):
        return F.softmax(self.fc(x), dim=1)

# Generate fake training data (like augmented embeddings)
np.random.seed(42)
torch.manual_seed(42)

n_samples = 3250  # 65 classes * 50 samples
train_embeddings = torch.randn(n_samples, 512)
train_embeddings = train_embeddings / train_embeddings.norm(dim=-1, keepdim=True)  # L2 normalize
train_labels = torch.arange(65).repeat_interleave(50)  # 50 samples per class

# Create model
model = SoftmaxNet()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Train for 1 epoch
dataset = TensorDataset(train_embeddings, train_labels)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

print(f"Training on {len(train_embeddings)} samples, {len(dataloader)} batches")

model.train()
total_loss = 0
total_correct = 0
total_samples = 0

for batch_idx, (x, y) in enumerate(dataloader):
    optimizer.zero_grad()
    outputs = model(x)
    loss = criterion(outputs, y)
    loss.backward()
    optimizer.step()

    total_loss += loss.item() * len(x)
    pred = outputs.argmax(dim=1)
    total_correct += (pred == y).sum().item()
    total_samples += len(x)

    if batch_idx % 50 == 0:
        print(f"Batch {batch_idx}/{len(dataloader)}: Loss={loss.item():.4f}")

train_loss = total_loss / total_samples
train_acc = total_correct / total_samples

print(f"\nAfter 1 epoch:")
print(f"  Training Loss: {train_loss:.4f}")
print(f"  Training Acc: {train_acc:.4f} ({train_acc*100:.2f}%)")

if train_acc < 0.10:
    print("  ❌ PROBLEM: Training accuracy is too low!")
else:
    print("  ✓ Training is working")

# ===== Test 2: Verify test evaluation =====
print("\n" + "="*80)
print("Test 2: Evaluation on held-out test set")
print("="*80)

# Generate test data (different from training)
test_embeddings = torch.randn(650, 512)  # 10 per class
test_embeddings = test_embeddings / test_embeddings.norm(dim=-1, keepdim=True)
test_labels = torch.arange(65).repeat_interleave(10)

model.eval()
with torch.no_grad():
    outputs = model(test_embeddings)
    pred = outputs.argmax(dim=1)
    test_correct = (pred == test_labels).sum().item()
    test_acc = test_correct / len(test_labels)

print(f"Test Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")

if test_acc < 0.05:
    print("  ❌ PROBLEM: Test accuracy near random! Model didn't generalize.")
    print("  Possible causes:")
    print("    1. Train/test data mismatch (different CLIP features?)")
    print("    2. Overfitting to augmented data")
    print("    3. Augmented data quality is poor")
elif test_acc > train_acc * 0.5:
    print("  ✓ Reasonable generalization")
else:
    print("  ⚠ Poor generalization (overfitting?)")

# ===== Test 3: Check prediction distribution =====
print("\n" + "="*80)
print("Test 3: Prediction distribution analysis")
print("="*80)

pred_counts = torch.bincount(pred, minlength=65)
print(f"Predictions per class (top 10):")
top_classes = pred_counts.argsort(descending=True)[:10]
for cls in top_classes:
    print(f"  Class {cls.item()}: {pred_counts[cls].item()} predictions ({pred_counts[cls].item()/len(pred)*100:.1f}%)")

# Check if model is biased
max_pred_ratio = pred_counts.max().item() / len(pred)
if max_pred_ratio > 0.5:
    print(f"  ❌ PROBLEM: Model is biased! {max_pred_ratio*100:.1f}% predictions go to one class")
else:
    print(f"  ✓ Predictions are distributed")

# ===== Test 4: Check output probabilities =====
print("\n" + "="*80)
print("Test 4: Output probability distribution")
print("="*80)

output_probs = outputs.cpu().numpy()
max_probs = output_probs.max(axis=1)
entropy = -(output_probs * np.log(output_probs + 1e-10)).sum(axis=1)

print(f"Max probability stats:")
print(f"  Mean: {max_probs.mean():.4f}")
print(f"  Std: {max_probs.std():.4f}")
print(f"  Min: {max_probs.min():.4f}")
print(f"  Max: {max_probs.max():.4f}")

print(f"\nEntropy stats (higher = more uncertain):")
print(f"  Mean: {entropy.mean():.4f}")
print(f"  Max possible: {np.log(65):.4f}")

if max_probs.mean() < 0.05:
    print("  ❌ PROBLEM: Probabilities are nearly uniform! Model is guessing randomly.")
elif entropy.mean() > np.log(65) * 0.9:
    print("  ❌ PROBLEM: Entropy is very high! Model is very uncertain.")
else:
    print("  ✓ Probabilities look reasonable")

# ===== Summary =====
print("\n" + "="*80)
print("SUMMARY")
print("="*80)

if train_acc > 0.5 and test_acc < 0.05:
    print("✓ Model CAN learn from augmented data")
    print("❌ But test accuracy is near random")
    print("\nLikely cause: Train/test data mismatch!")
    print("  → Check if CLIP features are extracted consistently")
    print("  → Verify test data is using the SAME CLIP model")
    print("  → Check if embeddings are normalized the same way")
elif train_acc < 0.1:
    print("❌ Model CANNOT learn from augmented data")
    print("\nLikely causes:")
    print("  → Augmented data quality is poor")
    print("  → Learning rate too small")
    print("  → Gradient flow issue")
else:
    print("✓ Training and testing both work in this simulation")
    print("⚠ Problem must be in actual GGEUR_Clip implementation")

print("\n" + "="*80)
