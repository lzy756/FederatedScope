#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
验证GGEUR归一化修复是否有效

测试修复后的GGEUR augmentation，确保增强数据已正确归一化
"""

import os
import sys
import torch
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

print("=" * 80)
print("验证GGEUR归一化修复")
print("=" * 80)

# 创建mock global geometries
torch.manual_seed(42)
np.random.seed(42)

dim = 512
n_components = 50

eigenvalues = torch.linspace(10, 0.1, n_components)
eigenvectors = torch.randn(dim, n_components)
eigenvectors, _ = torch.linalg.qr(eigenvectors)

global_geometries = {
    class_id: {
        'eigenvalues': eigenvalues.clone(),
        'eigenvectors': eigenvectors.clone()
    }
    for class_id in range(5)
}

# 导入GGEUR augmentation
from federatedscope.contrib.data_augmentation.ggeur_augmentation import GGEURAugmentation

# 创建augmenter
augmenter = GGEURAugmentation(
    global_geometries=global_geometries,
    scenario='single_domain',
    device='cpu',
    use_full_covariance=True,
    eigenvalue_scaling=True
)

print("\n[TEST 1] 测试单domain增强后的归一化")
print("-" * 80)

# 创建归一化的本地embeddings (模拟CLIP features)
local_embeddings = torch.randn(10, dim)
local_embeddings = local_embeddings / local_embeddings.norm(dim=-1, keepdim=True)
local_labels = torch.zeros(10, dtype=torch.long)

print(f"原始embeddings norm (应该≈1.0): {local_embeddings.norm(dim=-1).mean():.4f}")

# 增强
aug_embeddings, aug_labels = augmenter.augment_single_domain(
    local_embeddings,
    local_labels,
    class_id=0,
    N=20
)

# 检查增强后的norm
norms = aug_embeddings.norm(dim=-1)
print(f"增强后embeddings norm:")
print(f"  平均值: {norms.mean():.4f} (应该≈1.0)")
print(f"  最小值: {norms.min():.4f}")
print(f"  最大值: {norms.max():.4f}")
print(f"  标准差: {norms.std():.4f} (应该≈0.0)")

if torch.allclose(norms, torch.ones_like(norms), atol=1e-5):
    print("\n[PASS] 增强后的embeddings已正确归一化！")
else:
    print("\n[FAIL] 增强后的embeddings未归一化！")
    sys.exit(1)

print("\n[TEST 2] 测试多domain增强后的归一化")
print("-" * 80)

# 创建多domain augmenter
augmenter_multi = GGEURAugmentation(
    global_geometries=global_geometries,
    scenario='multi_domain',
    device='cpu',
    use_full_covariance=True,
    eigenvalue_scaling=True
)

# 创建其他domain的prototypes (也应该是归一化的)
other_domain_prototypes = {}
for i in range(1, 4):
    proto = torch.randn(dim)
    proto = proto / proto.norm()  # 归一化
    other_domain_prototypes[i] = proto

# 多domain增强
aug_embeddings_multi, aug_labels_multi = augmenter_multi.augment_multi_domain(
    local_embeddings,
    local_labels,
    class_id=0,
    other_domain_prototypes=other_domain_prototypes,
    N=10,
    M=50,
    step1_target=100
)

# 检查增强后的norm
norms_multi = aug_embeddings_multi.norm(dim=-1)
print(f"多domain增强后embeddings norm:")
print(f"  平均值: {norms_multi.mean():.4f} (应该≈1.0)")
print(f"  最小值: {norms_multi.min():.4f}")
print(f"  最大值: {norms_multi.max():.4f}")
print(f"  标准差: {norms_multi.std():.4f} (应该≈0.0)")

if torch.allclose(norms_multi, torch.ones_like(norms_multi), atol=1e-5):
    print("\n[PASS] 多domain增强后的embeddings已正确归一化！")
else:
    print("\n[FAIL] 多domain增强后的embeddings未归一化！")
    sys.exit(1)

print("\n[TEST 3] 测试完整dataset增强")
print("-" * 80)

# 创建包含多个类别的数据集
dataset_size = 100
dataset_embeddings = torch.randn(dataset_size, dim)
dataset_embeddings = dataset_embeddings / dataset_embeddings.norm(dim=-1, keepdim=True)
dataset_labels = torch.randint(0, 5, (dataset_size,))

print(f"原始数据集大小: {len(dataset_embeddings)}")
print(f"类别数: {len(torch.unique(dataset_labels))}")

# 完整数据集增强
aug_dataset_emb, aug_dataset_lab = augmenter.augment_dataset(
    dataset_embeddings,
    dataset_labels,
    N=10,
    target_total=500  # 每个类别500个样本
)

# 检查增强后的norm
norms_dataset = aug_dataset_emb.norm(dim=-1)
print(f"\n增强后数据集大小: {len(aug_dataset_emb)}")
print(f"增强后embeddings norm:")
print(f"  平均值: {norms_dataset.mean():.4f} (应该≈1.0)")
print(f"  最小值: {norms_dataset.min():.4f}")
print(f"  最大值: {norms_dataset.max():.4f}")
print(f"  标准差: {norms_dataset.std():.4f} (应该≈0.0)")

if torch.allclose(norms_dataset, torch.ones_like(norms_dataset), atol=1e-5):
    print("\n[PASS] 完整dataset增强后的embeddings已正确归一化！")
else:
    print("\n[FAIL] 完整dataset增强后的embeddings未归一化！")
    sys.exit(1)

print("\n" + "=" * 80)
print("所有测试通过！归一化修复有效！")
print("=" * 80)
print("\n修复说明:")
print("- 所有增强后的embeddings现在都已L2归一化 (norm=1.0)")
print("- 这与CLIP提取的特征保持一致")
print("- 训练数据和测试数据现在在相同的分布上")
print("- 应该能显著提高准确率！")
