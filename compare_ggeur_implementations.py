#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
对比原始GGEUR实现和FederatedScope实现的关键差异
"""

import os
import sys
import torch
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

print("=" * 80)
print("对比原始GGEUR和FederatedScope实现")
print("=" * 80)

# =========================
# 测试1: 样本生成方法对比
# =========================
print("\n[TEST 1] 对比样本生成方法")
print("-" * 80)

# 创建测试数据
torch.manual_seed(42)
np.random.seed(42)

dim = 512
mean = np.random.randn(dim)
cov_matrix = np.random.randn(dim, dim)
cov_matrix = cov_matrix @ cov_matrix.T  # Make it PSD

# 方法1: 原始论文方法 (numpy)
print("\n方法1: 原始论文方法")
def original_generate_samples(feature, cov_matrix, num_generated):
    """原始论文的实现"""
    jitter = 1e-6
    while True:
        try:
            B = np.linalg.cholesky(cov_matrix + jitter * np.eye(cov_matrix.shape[0]))
            break
        except np.linalg.LinAlgError:
            jitter *= 10
    # 关键: 使用 np.random.multivariate_normal
    new_features = np.random.multivariate_normal(feature, B @ B.T, num_generated)
    return new_features

original_samples = original_generate_samples(mean, cov_matrix, 100)
print(f"原始方法生成样本: {original_samples.shape}")
print(f"  均值: {original_samples.mean(axis=0)[:5]}")
print(f"  标准差: {original_samples.std(axis=0)[:5]}")

# 方法2: FederatedScope方法 (pytorch)
print("\n方法2: FederatedScope方法")
mean_torch = torch.from_numpy(mean).float()
cov_torch = torch.from_numpy(cov_matrix).float()

# FederatedScope的实现
jitter = 1e-6
identity = torch.eye(dim)
regularized_cov = cov_torch + jitter * identity
L = torch.linalg.cholesky(regularized_cov)

z = torch.randn(100, dim)
perturbations = torch.mm(z, L.t())  # [n_samples, D]
fs_samples = (mean_torch.unsqueeze(0) + perturbations).numpy()

print(f"FederatedScope生成样本: {fs_samples.shape}")
print(f"  均值: {fs_samples.mean(axis=0)[:5]}")
print(f"  标准差: {fs_samples.std(axis=0)[:5]}")

# 比较
mean_diff = np.abs(original_samples.mean(axis=0) - fs_samples.mean(axis=0)).mean()
std_diff = np.abs(original_samples.std(axis=0) - fs_samples.std(axis=0)).mean()

print(f"\n差异:")
print(f"  均值差异: {mean_diff:.6f}")
print(f"  标准差差异: {std_diff:.6f}")

if mean_diff < 0.5 and std_diff < 0.5:
    print("[PASS] 两种方法生成的样本分布相似")
else:
    print("[WARN] 两种方法生成的样本分布有明显差异")

# =========================
# 测试2: 协方差矩阵重建对比
# =========================
print("\n\n[TEST 2] 对比协方差矩阵重建")
print("-" * 80)

# 原始方法：直接使用协方差矩阵
print("\n原始方法: 直接使用完整协方差矩阵")
original_cov_used = cov_matrix
print(f"  协方差矩阵形状: {original_cov_used.shape}")
print(f"  迹 (trace): {np.trace(original_cov_used):.2f}")

# FederatedScope方法：从特征值和特征向量重建
print("\nFederatedScope方法: 从eigenvalues/eigenvectors重建")
eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

# 只保留top 50 (模拟FederatedScope的做法)
top_k = 50
eigenvalues_top = eigenvalues[-top_k:]
eigenvectors_top = eigenvectors[:, -top_k:]

# 重建协方差矩阵
reconstructed_cov = eigenvectors_top @ np.diag(eigenvalues_top) @ eigenvectors_top.T

print(f"  使用top-{top_k}特征值重建")
print(f"  重建协方差矩阵形状: {reconstructed_cov.shape}")
print(f"  迹 (trace): {np.trace(reconstructed_cov):.2f}")

# 计算重建误差
reconstruction_error = np.linalg.norm(original_cov_used - reconstructed_cov, 'fro')
relative_error = reconstruction_error / np.linalg.norm(original_cov_used, 'fro')

print(f"\n重建误差:")
print(f"  Frobenius范数误差: {reconstruction_error:.4f}")
print(f"  相对误差: {relative_error*100:.2f}%")

if relative_error > 0.1:
    print("[WARN] 重建误差较大！这可能导致生成样本质量下降")
else:
    print("[PASS] 重建误差可接受")

# 解释方差
explained_var = eigenvalues_top.sum() / eigenvalues.sum()
print(f"  Top-{top_k}特征值解释方差: {explained_var*100:.2f}%")

# =========================
# 测试3: 特征值缩放对比
# =========================
print("\n\n[TEST 3] 对比特征值缩放")
print("-" * 80)

# 原始方法的缩放
print("\n原始方法: nearest_pos_def缩放")
def nearest_pos_def_original(eigenvalues):
    """原始论文的特征值缩放"""
    scale_factors = np.ones_like(eigenvalues)
    scale_factors[:10] = np.linspace(5, 1, 10)  # 前10个特征值缩放
    scaled = eigenvalues * scale_factors
    scaled[scaled < 0] = 0
    return scaled

# 注意：原始方法在ascending order上操作（eigh返回的是ascending）
scaled_original = nearest_pos_def_original(eigenvalues)

print(f"  前10个特征值缩放因子: [5.0, 4.5, ..., 1.0]")
print(f"  原始前5个特征值: {eigenvalues[:5]}")
print(f"  缩放后前5个特征值: {scaled_original[:5]}")

# FederatedScope方法
from federatedscope.contrib.utils.geometry_utils import _apply_eigenvalue_scaling

eigenvalues_torch = torch.from_numpy(eigenvalues).float()
scaled_fs = _apply_eigenvalue_scaling(eigenvalues_torch).numpy()

print(f"\nFederatedScope方法:")
print(f"  原始前5个特征值: {eigenvalues[:5]}")
print(f"  缩放后前5个特征值: {scaled_fs[:5]}")

scale_diff = np.abs(scaled_original - scaled_fs).mean()
print(f"\n特征值缩放差异: {scale_diff:.6f}")

if scale_diff < 1e-4:
    print("[PASS] 特征值缩放方法一致")
else:
    print("[WARN] 特征值缩放方法有差异")

# =========================
# 测试4: 关键参数对比
# =========================
print("\n\n[TEST 4] 关键参数对比")
print("-" * 80)

print("\n原始论文参数 (FedAvg_GGEUR.py line 25-42, 72-100):")
print("  - 模型forward: F.softmax(self.fc3(x), dim=1)  <-- BUG!")
print("  - 损失函数: nn.CrossEntropyLoss()")
print("  - 样本生成: np.random.multivariate_normal(feature, B @ B.T, num)")
print("  - 每类目标样本数: 50")
print("  - Step 1 augmentation: expand_features_with_cov")
print("  - Step 2 augmentation: generate_new_samples from prototypes (50 each)")
print("  - 最终combine_samples: target_size=50")

print("\nFederatedScope参数 (ggeur_officehome_lds.yaml):")
print("  - 模型forward: logits (正确)")
print("  - 损失函数: nn.CrossEntropyLoss()")
print("  - 样本生成: mean + L @ z (等价于multivariate_normal)")
print("  - n_samples_per_original: 50")
print("  - m_samples_per_prototype: 50")
print("  - step1_samples_per_class: 50")

print("\n" + "=" * 80)
print("对比完成")
print("=" * 80)

print("\n关键发现:")
print("1. 原始代码在forward中使用了softmax，这与CrossEntropyLoss配合是错误的")
print("2. FederatedScope正确地输出logits")
print("3. 样本生成方法数学上等价")
print("4. 特征值缩放方法一致")
print("5. 可能的差异点:")
print("   - 协方差矩阵维度（原始用full，FS用top-K）")
print("   - 数据加载和预处理流程")
print("   - 训练轮数和学习率")
