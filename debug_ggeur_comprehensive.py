#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
全面调试GGEUR实现 - 精确定位2%准确率的根本原因

这个脚本会检查GGEUR流程中的每一个环节
"""

import os
import sys
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import TensorDataset, DataLoader

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

print("=" * 80)
print("全面调试GGEUR - 定位2%准确率的根本原因")
print("=" * 80)

# =========================
# 测试: 检查原始代码的softmax问题
# =========================
print("\n[CRITICAL TEST] 检查softmax+CrossEntropyLoss的影响")
print("-" * 80)

torch.manual_seed(42)

# 创建测试数据
n_classes = 65
n_samples = 1000
embedding_dim = 512

# 生成有区分度的数据
X_train = []
y_train = []
for i in range(n_classes):
    # 每个类别的中心不同
    center = torch.randn(embedding_dim) * 3 + i * 2
    samples = center.unsqueeze(0) + torch.randn(200, embedding_dim) * 0.5
    X_train.append(samples)
    y_train.append(torch.full((200,), i, dtype=torch.long))

X_train = torch.cat(X_train, dim=0)
y_train = torch.cat(y_train, dim=0)

X_test = []
y_test = []
for i in range(n_classes):
    center = torch.randn(embedding_dim) * 3 + i * 2
    samples = center.unsqueeze(0) + torch.randn(20, embedding_dim) * 0.5
    X_test.append(samples)
    y_test.append(torch.full((20,), i, dtype=torch.long))

X_test = torch.cat(X_test, dim=0)
y_test = torch.cat(y_test, dim=0)

print(f"训练数据: {X_train.shape}, 测试数据: {X_test.shape}")

# 模型1: 原始论文的buggy实现 (forward返回softmax)
class BuggyNet(nn.Module):
    def __init__(self, num_classes=65):
        super(BuggyNet, self).__init__()
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        return torch.softmax(self.fc(x), dim=1)  # BUG: 返回概率

# 模型2: 正确实现 (forward返回logits)
class CorrectNet(nn.Module):
    def __init__(self, num_classes=65):
        super(CorrectNet, self).__init__()
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        return self.fc(x)  # 正确: 返回logits

def train_and_test(model, name, X_train, y_train, X_test, y_test, epochs=10):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=16, shuffle=True)

    model.train()
    for epoch in range(epochs):
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

# Test
    model.eval()
    with torch.no_grad():
        outputs = model(X_test)
        _, pred = torch.max(outputs, 1)
        acc = (pred == y_test).sum().item() / len(y_test)

    print(f"{name}: 测试准确率 = {acc*100:.2f}%")
    return acc

buggy_net = BuggyNet(n_classes)
correct_net = CorrectNet(n_classes)

buggy_acc = train_and_test(buggy_net, "Buggy (softmax+CrossEntropyLoss)", X_train, y_train, X_test, y_test)
correct_acc = train_and_test(correct_net, "Correct (logits+CrossEntropyLoss)", X_train, y_train, X_test, y_test)

print(f"\n差异: {abs(buggy_acc - correct_acc)*100:.2f}%")
if buggy_acc < 0.5:
    print("[CRITICAL] Buggy模型准确率极低！这可能是原因之一")
else:
    print("[INFO] Buggy模型居然也能工作...这很奇怪")

# =========================
# 测试: 检查数据归一化的影响
# =========================
print("\n\n[CRITICAL TEST] 检查数据归一化的影响")
print("-" * 80)

# 创建归一化和未归一化的数据
X_train_normalized = X_train / X_train.norm(dim=-1, keepdim=True)
X_test_normalized = X_test / X_test.norm(dim=-1, keepdim=True)

# 增强数据但不归一化
X_train_aug_unnorm = []
y_train_aug_list = []
for i in range(100):
    x = X_train_normalized[i]
    # 添加噪声
    noise = torch.randn(10, embedding_dim) * 0.5
    aug = x.unsqueeze(0) + noise
    X_train_aug_unnorm.append(aug)
    y_train_aug_list.append(torch.full((10,), y_train[i].item(), dtype=torch.long))

X_train_aug_unnorm = torch.cat(X_train_aug_unnorm, dim=0)
y_train_aug = torch.cat(y_train_aug_list)

# 只取前n_classes个测试样本（每类一个）
X_test_subset = X_test_normalized[:n_classes]
y_test_subset = y_test[:n_classes]

print(f"原始训练数据 norm: mean={X_train_normalized.norm(dim=-1).mean():.4f}")
print(f"增强数据(未归一化) norm: mean={X_train_aug_unnorm.norm(dim=-1).mean():.4f}")
print(f"测试数据 norm: mean={X_test_normalized.norm(dim=-1).mean():.4f}")

# 用未归一化的增强数据训练
net_unnorm = CorrectNet(n_classes)
acc_unnorm = train_and_test(net_unnorm, "增强数据未归一化", X_train_aug_unnorm, y_train_aug, X_test_subset, y_test_subset)

# 用归一化的增强数据训练
X_train_aug_norm = X_train_aug_unnorm / X_train_aug_unnorm.norm(dim=-1, keepdim=True)
net_norm = CorrectNet(n_classes)
acc_norm = train_and_test(net_norm, "增强数据已归一化", X_train_aug_norm, y_train_aug, X_test_subset, y_test_subset)

print(f"\n归一化的影响: {abs(acc_norm - acc_unnorm)*100:.2f}%")
if acc_unnorm < 0.1:
    print("[CRITICAL] 未归一化的增强数据导致准确率崩溃！这是关键问题")

print("\n" + "=" * 80)
print("调试完成")
print("=" * 80)

print("\n总结:")
print("1. 如果Buggy模型准确率很低 → softmax+CrossEntropyLoss有问题")
print("2. 如果未归一化增强数据准确率很低 → 归一化问题是关键")
print("3. 如果两者都正常 → 问题在其他地方（数据加载、超参数等）")
