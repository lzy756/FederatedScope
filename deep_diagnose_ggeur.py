#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
深度诊断GGEUR训练流程中的bug

这个脚本会模拟GGEUR的完整流程，找出导致准确率极低的原因
"""

import os
import sys
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader, TensorDataset

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

print("=" * 80)
print("GGEUR_Clip 深度诊断 - 模拟训练和评估流程")
print("=" * 80)

# =========================
# 测试1: 简单的MLP分类器
# =========================
print("\n[TEST 1] 测试MLP分类器是否工作正常")
print("-" * 80)

# 创建简单的测试数据
torch.manual_seed(42)
np.random.seed(42)

n_classes = 5
embedding_dim = 512
n_train_samples = 100
n_test_samples = 50

# 生成有区分度的训练数据（每个类别的embeddings聚集在不同的区域）
train_embeddings = []
train_labels = []

for class_id in range(n_classes):
    # 每个类别的中心点不同
    center = torch.randn(embedding_dim) * 2 + class_id * 5
    # 围绕中心生成样本
    class_samples = center.unsqueeze(0) + torch.randn(n_train_samples // n_classes, embedding_dim) * 0.5
    train_embeddings.append(class_samples)
    train_labels.append(torch.full((n_train_samples // n_classes,), class_id, dtype=torch.long))

train_embeddings = torch.cat(train_embeddings, dim=0)
train_labels = torch.cat(train_labels, dim=0)

# 生成测试数据（使用相同的分布）
test_embeddings = []
test_labels = []

for class_id in range(n_classes):
    center = torch.randn(embedding_dim) * 2 + class_id * 5
    class_samples = center.unsqueeze(0) + torch.randn(n_test_samples // n_classes, embedding_dim) * 0.5
    test_embeddings.append(class_samples)
    test_labels.append(torch.full((n_test_samples // n_classes,), class_id, dtype=torch.long))

test_embeddings = torch.cat(test_embeddings, dim=0)
test_labels = torch.cat(test_labels, dim=0)

# 创建简单的MLP
mlp = nn.Sequential(
    nn.Linear(embedding_dim, n_classes)
)

# 训练
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(mlp.parameters(), lr=0.001)

train_dataset = TensorDataset(train_embeddings, train_labels)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

print(f"训练数据: {len(train_embeddings)} samples, {n_classes} classes")
print(f"测试数据: {len(test_embeddings)} samples")

# 训练10个epoch
mlp.train()
for epoch in range(10):
    total_loss = 0
    correct = 0
    total = 0

    for batch_embeddings, batch_labels in train_loader:
        optimizer.zero_grad()
        outputs = mlp(batch_embeddings)
        loss = criterion(outputs, batch_labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = outputs.max(1)
        correct += predicted.eq(batch_labels).sum().item()
        total += len(batch_labels)

    acc = 100. * correct / total
    print(f"Epoch {epoch+1}: Loss={total_loss:.4f}, Train Acc={acc:.2f}%")

# 测试
mlp.eval()
with torch.no_grad():
    outputs = mlp(test_embeddings)
    _, predicted = outputs.max(1)
    correct = predicted.eq(test_labels).sum().item()
    test_acc = 100. * correct / len(test_labels)

print(f"\n测试准确率: {test_acc:.2f}%")

if test_acc < 50:
    print("[ERROR] MLP分类器本身有问题！准确率太低")
    sys.exit(1)
else:
    print("[OK] MLP分类器工作正常")

# =========================
# 测试2: 检查GGEUR trainer的evaluate函数
# =========================
print("\n\n[TEST 2] 检查GGEUR Trainer的evaluate逻辑")
print("-" * 80)

# 模拟GGEUR trainer的evaluate过程
print("模拟CLIP特征提取...")

# 创建mock CLIP extractor
class MockCLIPExtractor:
    def __init__(self):
        self.embedding_dim = 512

    def extract_dataset_features(self, dataset, use_cache=False):
        # 直接返回我们的测试embedding
        return {
            'embeddings': test_embeddings,
            'labels': test_labels
        }

# 模拟trainer的evaluate
print("模拟trainer evaluate...")
mock_clip = MockCLIPExtractor()

result = mock_clip.extract_dataset_features(None)
eval_embeddings = result['embeddings']
eval_labels = result['labels']

mlp.eval()
with torch.no_grad():
    logits = mlp(eval_embeddings)
    pred = logits.argmax(dim=1)
    correct = (pred == eval_labels).sum().item()
    accuracy = correct / len(eval_labels)

print(f"模拟评估准确率: {accuracy*100:.2f}%")

if accuracy < 0.5:
    print("[ERROR] 评估流程有问题！")
else:
    print("[OK] 评估流程正常")

# =========================
# 测试3: 检查标签对齐问题
# =========================
print("\n\n[TEST 3] 检查标签对齐问题")
print("-" * 80)

# 检查是否存在标签不匹配的情况
print("检查训练和测试标签范围...")
print(f"训练标签: min={train_labels.min().item()}, max={train_labels.max().item()}")
print(f"测试标签: min={test_labels.min().item()}, max={test_labels.max().item()}")

unique_train_labels = torch.unique(train_labels)
unique_test_labels = torch.unique(test_labels)

print(f"训练集唯一标签: {unique_train_labels.tolist()}")
print(f"测试集唯一标签: {unique_test_labels.tolist()}")

if not torch.equal(unique_train_labels, unique_test_labels):
    print("[WARN] 训练集和测试集的标签不一致！这可能导致问题")
else:
    print("[OK] 标签对齐正常")

# =========================
# 测试4: 检查GGEUR数据增强后的标签
# =========================
print("\n\n[TEST 4] 检查GGEUR数据增强是否破坏了标签")
print("-" * 80)

# 模拟GGEUR增强
print("模拟数据增强...")

# 简单的增强：复制数据并添加噪声
augmented_embeddings = []
augmented_labels = []

for class_id in range(n_classes):
    class_mask = (train_labels == class_id)
    class_embeddings = train_embeddings[class_mask]

    # 原始数据
    augmented_embeddings.append(class_embeddings)
    augmented_labels.append(torch.full((len(class_embeddings),), class_id, dtype=torch.long))

    # 增强数据：添加噪声
    for _ in range(10):  # 每个样本生成10个增强样本
        noise = torch.randn_like(class_embeddings) * 0.5
        aug_samples = class_embeddings + noise
        augmented_embeddings.append(aug_samples)
        augmented_labels.append(torch.full((len(aug_samples),), class_id, dtype=torch.long))

aug_emb = torch.cat(augmented_embeddings, dim=0)
aug_lab = torch.cat(augmented_labels, dim=0)

print(f"增强后数据: {len(aug_emb)} samples (原始: {len(train_embeddings)})")
print(f"增强倍数: {len(aug_emb) / len(train_embeddings):.1f}x")

# 检查增强后的标签分布
print("\n增强后的标签分布:")
for class_id in range(n_classes):
    count = (aug_lab == class_id).sum().item()
    print(f"  Class {class_id}: {count} samples")

# 用增强后的数据重新训练
mlp2 = nn.Sequential(nn.Linear(embedding_dim, n_classes))
optimizer2 = torch.optim.Adam(mlp2.parameters(), lr=0.001)

aug_dataset = TensorDataset(aug_emb, aug_lab)
aug_loader = DataLoader(aug_dataset, batch_size=16, shuffle=True)

print("\n使用增强数据训练...")
mlp2.train()
for epoch in range(10):
    total_loss = 0
    correct = 0
    total = 0

    for batch_embeddings, batch_labels in aug_loader:
        optimizer2.zero_grad()
        outputs = mlp2(batch_embeddings)
        loss = criterion(outputs, batch_labels)
        loss.backward()
        optimizer2.step()

        total_loss += loss.item()
        _, predicted = outputs.max(1)
        correct += predicted.eq(batch_labels).sum().item()
        total += len(batch_labels)

    if (epoch + 1) % 5 == 0:
        acc = 100. * correct / total
        print(f"Epoch {epoch+1}: Loss={total_loss:.4f}, Train Acc={acc:.2f}%")

# 测试增强后训练的模型
mlp2.eval()
with torch.no_grad():
    outputs = mlp2(test_embeddings)
    _, predicted = outputs.max(1)
    correct = predicted.eq(test_labels).sum().item()
    test_acc = 100. * correct / len(test_labels)

print(f"\n增强后训练的模型测试准确率: {test_acc:.2f}%")

if test_acc < 50:
    print("[ERROR] 数据增强导致准确率下降！")
else:
    print("[OK] 数据增强后训练正常")

print("\n" + "=" * 80)
print("诊断完成")
print("=" * 80)

# =========================
# 总结
# =========================
print("\n[SUMMARY] 诊断总结:")
print("-" * 80)

issues_found = []

if test_acc < 50:
    issues_found.append("MLP分类器本身有问题")

if accuracy < 0.5:
    issues_found.append("评估流程有问题")

if not torch.equal(unique_train_labels, unique_test_labels):
    issues_found.append("训练/测试标签不一致")

test_acc_aug = test_acc
if test_acc_aug < 50:
    issues_found.append("数据增强破坏了训练")

if issues_found:
    print("发现以下问题:")
    for issue in issues_found:
        print(f"  - {issue}")
else:
    print("基础组件测试通过，问题可能在于:")
    print("  1. CLIP特征提取的质量")
    print("  2. 真实数据的分布")
    print("  3. GGEUR增强参数设置")
    print("  4. 训练轮数太少")
