#!/usr/bin/env python
"""
GGEUR独立运行脚本 - 完全匹配论文实现

可直接在服务器上运行，无需复杂的FederatedScope配置。

用法:
    # 使用prepare_ggeur_offline_data.py生成的预处理数据
    python run_ggeur_standalone.py --data_dir /path/to/preprocessed_data

    # 使用论文原始预计算数据
    python run_ggeur_standalone.py --data_dir /path/to/paper/data --use_paper_format

    # 自定义参数
    python run_ggeur_standalone.py --data_dir /path/to/data --num_rounds 100 --lr 0.001
"""

import os
import sys
import argparse
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# 论文核心算法 - 精确复制
# ============================================================================

def nearest_pos_def(cov_matrix: np.ndarray) -> np.ndarray:
    """
    确保协方差矩阵正定 - 完全匹配论文

    论文代码: prototype_cov_matrix_generate_features.py
    """
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

    # 对最小的10个特征值进行缩放
    scale_factors = np.ones_like(eigenvalues)
    scale_factors[:10] = np.linspace(5, 0.5, 10)

    eigenvalues = eigenvalues * scale_factors
    eigenvalues[eigenvalues < 0] = 0

    return eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T


def generate_samples(feature: np.ndarray, cov_matrix: np.ndarray,
                    num_samples: int) -> np.ndarray:
    """生成新样本 - 完全匹配论文"""
    cov_matrix = nearest_pos_def(cov_matrix)

    jitter = 1e-6
    for _ in range(10):
        try:
            B = np.linalg.cholesky(cov_matrix + jitter * np.eye(512))
            break
        except np.linalg.LinAlgError:
            jitter *= 10

    return np.random.multivariate_normal(feature, B @ B.T, num_samples)


def augment_features(original: np.ndarray, cov: np.ndarray,
                    other_protos: List[np.ndarray],
                    target_size: int = 50, n_per_sample: int = 50) -> np.ndarray:
    """数据增强 - 完全匹配论文"""
    num_original = len(original) if len(original.shape) > 0 else 0

    if num_original >= target_size:
        indices = np.random.choice(num_original, target_size, replace=False)
        return original[indices]

    # Step 1: 从原始样本扩展
    if num_original > 0:
        all_gen = [generate_samples(f, cov, n_per_sample) for f in original]
        all_gen = np.vstack(all_gen)
        expanded = all_gen[np.random.choice(len(all_gen), min(50, len(all_gen)), replace=False)]
    else:
        expanded = np.empty((0, 512))

    # Step 2: 从其他域原型生成
    other_gen = []
    for proto in other_protos:
        if proto is not None and len(proto) > 0:
            other_gen.append(generate_samples(proto, cov, 50))
    other_gen = np.vstack(other_gen) if other_gen else np.empty((0, 512))

    # Step 3: 组合
    if num_original == 0:
        if len(other_gen) >= target_size:
            return other_gen[np.random.choice(len(other_gen), target_size, replace=False)]
        return other_gen

    num_needed = target_size - num_original
    all_gen_list = [x for x in [expanded, other_gen] if len(x) > 0]
    all_gen = np.vstack(all_gen_list) if all_gen_list else np.empty((0, 512))

    if len(all_gen) >= num_needed:
        selected = all_gen[np.random.choice(len(all_gen), num_needed, replace=False)]
        return np.vstack([original, selected])
    elif len(all_gen) > 0:
        return np.vstack([original, all_gen])
    return original


# ============================================================================
# 模型和训练
# ============================================================================

class FeatureDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.from_numpy(features).float() if isinstance(features, np.ndarray) else features.float()
        self.labels = torch.from_numpy(labels).long() if isinstance(labels, np.ndarray) else labels.long()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


class GGEURNet(nn.Module):
    """论文模型 - 带softmax"""
    def __init__(self, num_classes=65):
        super().__init__()
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        return F.softmax(self.fc(x), dim=1)


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    loss_sum, correct, total = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        loss_sum += loss.item() * x.size(0)
        correct += (out.argmax(1) == y).sum().item()
        total += y.size(0)
    return loss_sum / total, correct / total


def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            correct += (model(x).argmax(1) == y).sum().item()
            total += y.size(0)
    return correct / total


def fedavg(global_model, client_models):
    state = global_model.state_dict()
    for k in state:
        state[k] = torch.stack([m.state_dict()[k].float() for m in client_models]).mean(0)
    global_model.load_state_dict(state)
    return global_model


# ============================================================================
# 主函数
# ============================================================================

def run_with_paper_data(args):
    """使用论文预计算数据运行"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"设备: {device}")

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    # 路径
    train_dir = os.path.join(args.paper_data_root, 'backup_vit_b32')
    test_dir = os.path.join(args.paper_data_root, 'clip_office_home_test_features')
    cov_dir = os.path.join(args.paper_data_root, 'cov_matrix_output')

    domains = ['Art', 'Clipart', 'Product', 'Real_World']
    domain2id = {'Art': 0, 'Clipart': 1, 'Product': 2, 'Real_World': 3}
    num_classes = 65

    # 1. 加载数据
    logger.info("="*60)
    logger.info("Step 1: 加载数据")
    logger.info("="*60)

    client_features = {}
    for domain in domains:
        cid = domain2id[domain]
        features_by_class = {}
        for c in range(num_classes):
            f = os.path.join(train_dir, domain, f'client_{cid}_class_{c}_original_features.npy')
            if os.path.exists(f):
                feat = np.load(f)
                if len(feat) > 0:
                    features_by_class[c] = feat
        client_features[cid] = features_by_class
        total = sum(len(v) for v in features_by_class.values())
        logger.info(f"  {domain} (Client {cid}): {total} 训练样本, {len(features_by_class)} 类别")

    # 加载协方差
    logger.info("\n加载协方差矩阵...")
    covs = {}
    for c in range(num_classes):
        f = os.path.join(cov_dir, f'class_{c}_cov_matrix.npy')
        covs[c] = np.load(f) if os.path.exists(f) else np.zeros((512, 512))
    logger.info(f"  加载 {len(covs)} 个协方差矩阵")

    # 加载测试
    logger.info("\n加载测试数据...")
    test_feats, test_labels = [], []
    for domain in domains:
        f = os.path.join(test_dir, domain, f'{domain}_test_features.npy')
        l = os.path.join(test_dir, domain, f'{domain}_test_labels.npy')
        if os.path.exists(f):
            test_feats.append(np.load(f))
            test_labels.append(np.load(l))
            logger.info(f"  {domain}: {len(np.load(f))} 测试样本")
    test_feats = np.vstack(test_feats)
    test_labels = np.concatenate(test_labels)
    logger.info(f"  测试集总计: {len(test_feats)} 样本")

    # 2. 计算原型
    logger.info("\n" + "="*60)
    logger.info("Step 2: 计算原型")
    logger.info("="*60)
    protos = {}
    for cid, feat_by_class in client_features.items():
        protos[cid] = {}
        for c, feat in feat_by_class.items():
            protos[cid][c] = np.mean(feat, axis=0)
        logger.info(f"  Client {cid}: {len(protos[cid])} 个原型")

    # 3. 数据增强
    logger.info("\n" + "="*60)
    logger.info("Step 3: 离线数据增强")
    logger.info("="*60)
    logger.info(f"  target_size={args.target_size}, n_per_sample={args.n_per_sample}")

    augmented = {}
    for cid in sorted(client_features.keys()):
        all_f, all_l = [], []
        other_cids = [x for x in client_features if x != cid]

        for c in range(num_classes):
            orig = client_features[cid].get(c, np.empty((0, 512)))
            if len(orig.shape) == 1:
                orig = np.empty((0, 512))
            cov = covs.get(c, np.zeros((512, 512)))
            other_p = [protos[oid].get(c) for oid in other_cids if c in protos[oid]]

            if np.all(cov == 0):
                if len(orig) > 0:
                    all_f.append(orig)
                    all_l.append(np.full(len(orig), c, dtype=np.int64))
                continue

            aug = augment_features(orig, cov, other_p, args.target_size, args.n_per_sample)
            if len(aug) > 0:
                all_f.append(aug)
                all_l.append(np.full(len(aug), c, dtype=np.int64))

        if all_f:
            augmented[cid] = (np.vstack(all_f), np.concatenate(all_l))
            logger.info(f"  Client {cid}: {len(augmented[cid][0])} 增强样本")

    # 4. 联邦训练
    logger.info("\n" + "="*60)
    logger.info("Step 4: 联邦训练 (FedAvg)")
    logger.info("="*60)
    logger.info(f"  num_rounds={args.num_rounds}")
    logger.info(f"  local_epochs={args.local_epochs}")
    logger.info(f"  batch_size={args.batch_size}")
    logger.info(f"  lr={args.lr}")

    global_model = GGEURNet(num_classes).to(device)
    client_models = [GGEURNet(num_classes).to(device) for _ in augmented]
    criterion = nn.CrossEntropyLoss()

    loaders = [DataLoader(FeatureDataset(*augmented[cid]), args.batch_size, shuffle=True)
               for cid in sorted(augmented)]
    test_loader = DataLoader(FeatureDataset(test_feats, test_labels), args.batch_size, shuffle=False)

    best_acc = 0
    logger.info("\n开始训练...")
    for r in range(args.num_rounds):
        # 分发模型
        for m in client_models:
            m.load_state_dict(global_model.state_dict())

        # 本地训练
        for i, loader in enumerate(loaders):
            opt = optim.Adam(client_models[i].parameters(), lr=args.lr)
            for _ in range(args.local_epochs):
                train_epoch(client_models[i], loader, opt, criterion, device)

        # FedAvg聚合
        global_model = fedavg(global_model, client_models)

        # 评估
        acc = evaluate(global_model, test_loader, device)
        best_acc = max(best_acc, acc)

        if (r + 1) % 10 == 0 or r == 0:
            logger.info(f"  Round {r+1:3d}/{args.num_rounds}: "
                       f"Test Acc = {acc:.4f} ({acc*100:.2f}%), "
                       f"Best = {best_acc:.4f} ({best_acc*100:.2f}%)")

    # 最终结果
    logger.info("\n" + "="*60)
    logger.info("最终结果")
    logger.info("="*60)
    logger.info(f"  最佳测试精度: {best_acc:.4f} ({best_acc*100:.2f}%)")

    # 每域精度
    logger.info("\n  每域精度:")
    for domain in domains:
        f = os.path.join(test_dir, domain, f'{domain}_test_features.npy')
        l = os.path.join(test_dir, domain, f'{domain}_test_labels.npy')
        if os.path.exists(f):
            loader = DataLoader(FeatureDataset(np.load(f), np.load(l)), args.batch_size)
            acc = evaluate(global_model, loader, device)
            logger.info(f"    {domain}: {acc:.4f} ({acc*100:.2f}%)")

    return best_acc


def run_with_preprocessed_data(args):
    """使用prepare_ggeur_offline_data.py生成的数据运行"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"设备: {device}")

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    # 加载元数据
    import json
    metadata_file = os.path.join(args.data_dir, 'metadata.json')
    if os.path.exists(metadata_file):
        with open(metadata_file) as f:
            metadata = json.load(f)
        domains = metadata['domains']
        num_classes = metadata['num_classes']
        logger.info(f"从metadata加载: {len(domains)} 个域, {num_classes} 个类别")
    else:
        # 默认Office-Home
        domains = ['Art', 'Clipart', 'Product', 'Real_World']
        num_classes = 65

    train_dir = os.path.join(args.data_dir, 'train_features')
    test_dir = os.path.join(args.data_dir, 'test_features')
    cov_dir = os.path.join(args.data_dir, 'cov_matrices')

    # 加载数据
    logger.info("="*60)
    logger.info("Step 1: 加载数据")
    logger.info("="*60)

    client_features = {}
    for domain_idx, domain in enumerate(domains):
        features_by_class = {}
        domain_dir = os.path.join(train_dir, domain)
        if not os.path.exists(domain_dir):
            continue
        for c in range(num_classes):
            f = os.path.join(domain_dir, f'client_{domain_idx}_class_{c}_original_features.npy')
            if os.path.exists(f):
                feat = np.load(f)
                if len(feat) > 0:
                    features_by_class[c] = feat
        client_features[domain_idx] = features_by_class
        total = sum(len(v) for v in features_by_class.values())
        logger.info(f"  {domain} (Client {domain_idx}): {total} 训练样本")

    # 加载协方差
    covs = {}
    for c in range(num_classes):
        f = os.path.join(cov_dir, f'class_{c}_cov_matrix.npy')
        covs[c] = np.load(f) if os.path.exists(f) else np.zeros((512, 512))

    # 加载测试
    test_feats, test_labels = [], []
    for domain in domains:
        f = os.path.join(test_dir, domain, f'{domain}_test_features.npy')
        l = os.path.join(test_dir, domain, f'{domain}_test_labels.npy')
        if os.path.exists(f):
            test_feats.append(np.load(f))
            test_labels.append(np.load(l))
    test_feats = np.vstack(test_feats)
    test_labels = np.concatenate(test_labels)
    logger.info(f"  测试集: {len(test_feats)} 样本")

    # 计算原型
    protos = {}
    for cid, feat_by_class in client_features.items():
        protos[cid] = {c: np.mean(feat, axis=0) for c, feat in feat_by_class.items()}

    # 数据增强
    logger.info("\nStep 2: 数据增强")
    augmented = {}
    for cid in sorted(client_features.keys()):
        all_f, all_l = [], []
        other_cids = [x for x in client_features if x != cid]
        for c in range(num_classes):
            orig = client_features[cid].get(c, np.empty((0, 512)))
            if len(orig.shape) == 1:
                orig = np.empty((0, 512))
            cov = covs.get(c, np.zeros((512, 512)))
            other_p = [protos[oid].get(c) for oid in other_cids if c in protos[oid]]
            if np.all(cov == 0):
                if len(orig) > 0:
                    all_f.append(orig)
                    all_l.append(np.full(len(orig), c, dtype=np.int64))
                continue
            aug = augment_features(orig, cov, other_p, args.target_size, args.n_per_sample)
            if len(aug) > 0:
                all_f.append(aug)
                all_l.append(np.full(len(aug), c, dtype=np.int64))
        if all_f:
            augmented[cid] = (np.vstack(all_f), np.concatenate(all_l))
            logger.info(f"  Client {cid}: {len(augmented[cid][0])} 增强样本")

    # 联邦训练
    logger.info("\nStep 3: 联邦训练")
    global_model = GGEURNet(num_classes).to(device)
    client_models = [GGEURNet(num_classes).to(device) for _ in augmented]
    criterion = nn.CrossEntropyLoss()

    loaders = [DataLoader(FeatureDataset(*augmented[cid]), args.batch_size, shuffle=True)
               for cid in sorted(augmented)]
    test_loader = DataLoader(FeatureDataset(test_feats, test_labels), args.batch_size, shuffle=False)

    best_acc = 0
    for r in range(args.num_rounds):
        for m in client_models:
            m.load_state_dict(global_model.state_dict())
        for i, loader in enumerate(loaders):
            opt = optim.Adam(client_models[i].parameters(), lr=args.lr)
            for _ in range(args.local_epochs):
                train_epoch(client_models[i], loader, opt, criterion, device)
        global_model = fedavg(global_model, client_models)
        acc = evaluate(global_model, test_loader, device)
        best_acc = max(best_acc, acc)
        if (r + 1) % 10 == 0 or r == 0:
            logger.info(f"  Round {r+1}/{args.num_rounds}: Acc={acc:.4f}, Best={best_acc:.4f}")

    logger.info(f"\n最终结果: {best_acc:.4f} ({best_acc*100:.2f}%)")
    return best_acc


def main():
    parser = argparse.ArgumentParser(description='GGEUR独立运行脚本 - 完全匹配论文')

    # 数据路径
    parser.add_argument('--data_dir', type=str, required=True,
                       help='数据目录 (prepare_ggeur_offline_data.py输出目录 或 论文数据目录)')
    parser.add_argument('--use_paper_format', action='store_true',
                       help='使用论文原始数据格式 (backup_vit_b32等目录结构)')

    # 增强参数 (论文默认值)
    parser.add_argument('--target_size', type=int, default=50,
                       help='每类目标样本数 (默认: 50)')
    parser.add_argument('--n_per_sample', type=int, default=50,
                       help='每个原始样本生成的样本数 (默认: 50)')

    # 训练参数 (论文默认值)
    parser.add_argument('--num_rounds', type=int, default=50,
                       help='联邦训练轮数 (默认: 50)')
    parser.add_argument('--local_epochs', type=int, default=1,
                       help='每轮本地训练epochs (默认: 1)')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='批量大小 (默认: 16)')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='学习率 (默认: 0.001)')
    parser.add_argument('--seed', type=int, default=12345,
                       help='随机种子 (默认: 12345)')

    args = parser.parse_args()

    if not os.path.exists(args.data_dir):
        logger.error(f"数据目录不存在: {args.data_dir}")
        sys.exit(1)

    if args.use_paper_format:
        # 论文原始格式
        args.paper_data_root = args.data_dir
        run_with_paper_data(args)
    else:
        # 我们预处理脚本生成的格式
        run_with_preprocessed_data(args)


if __name__ == '__main__':
    main()
