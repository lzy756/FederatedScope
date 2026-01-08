"""
GGEUR_Clip FederatedScope运行脚本 - 使用论文预计算数据

这个脚本可以:
1. 使用论文预计算的CLIP特征和协方差矩阵
2. 执行离线增强
3. 运行标准FedAvg训练

完全匹配论文实现。
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

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 导入预处理模块
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from federatedscope.contrib.ggeur_preprocessing import (
    GGEURPreprocessor,
    aggregate_covariance_matrices,
    augment_class_features
)

# ============================================================================
# 配置
# ============================================================================

# 论文数据路径
PAPER_DATA_ROOT = "D:/Projects/2025CVPR_GGEUR/Multi Domain/Office-Home-LDS"
TRAIN_FEATURES_DIR = os.path.join(PAPER_DATA_ROOT, "backup_vit_b32")
TEST_FEATURES_DIR = os.path.join(PAPER_DATA_ROOT, "clip_office_home_test_features")
COV_MATRIX_DIR = os.path.join(PAPER_DATA_ROOT, "cov_matrix_output")

DOMAINS = ['Art', 'Clipart', 'Product', 'Real_World']
DOMAIN_TO_CLIENT = {'Art': 0, 'Clipart': 1, 'Product': 2, 'Real_World': 3}
NUM_CLASSES = 65


# ============================================================================
# Dataset
# ============================================================================

class FeatureDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.from_numpy(features).float() if isinstance(features, np.ndarray) else features.float()
        self.labels = torch.from_numpy(labels).long() if isinstance(labels, np.ndarray) else labels.long()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


# ============================================================================
# 模型 - 完全匹配论文
# ============================================================================

class MyNet(nn.Module):
    """论文的MyNet - 带softmax输出"""
    def __init__(self, num_classes=65):
        super().__init__()
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        return F.softmax(self.fc(x), dim=1)


# ============================================================================
# 数据加载
# ============================================================================

def load_train_features(domain):
    """加载训练特征"""
    domain_dir = os.path.join(TRAIN_FEATURES_DIR, domain)
    client_id = DOMAIN_TO_CLIENT[domain]

    features_by_class = {}

    for class_id in range(NUM_CLASSES):
        feat_file = os.path.join(domain_dir, f"client_{client_id}_class_{class_id}_original_features.npy")
        if os.path.exists(feat_file):
            features = np.load(feat_file)
            if len(features) > 0:
                features_by_class[class_id] = features

    return features_by_class


def load_test_features(domain):
    """加载测试特征"""
    feat_file = os.path.join(TEST_FEATURES_DIR, domain, f"{domain}_test_features.npy")
    label_file = os.path.join(TEST_FEATURES_DIR, domain, f"{domain}_test_labels.npy")

    if os.path.exists(feat_file) and os.path.exists(label_file):
        return np.load(feat_file), np.load(label_file)
    return None, None


def load_covariance_matrices():
    """加载预计算的协方差矩阵"""
    cov_matrices = {}
    for class_id in range(NUM_CLASSES):
        cov_file = os.path.join(COV_MATRIX_DIR, f"class_{class_id}_cov_matrix.npy")
        if os.path.exists(cov_file):
            cov_matrices[class_id] = np.load(cov_file)
        else:
            cov_matrices[class_id] = np.zeros((512, 512))
    return cov_matrices


# ============================================================================
# 训练函数 - 完全匹配论文
# ============================================================================

def train_epoch(model, dataloader, optimizer, criterion, device):
    """训练一个epoch"""
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)  # 概率输出
        loss = criterion(outputs, labels)  # 论文的方式
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    return running_loss / total, correct / total


def evaluate(model, dataloader, device):
    """评估模型"""
    model.eval()
    correct, total = 0, 0

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return correct / total


def federated_averaging(global_model, client_models):
    """FedAvg聚合"""
    global_dict = global_model.state_dict()
    for k in global_dict.keys():
        global_dict[k] = torch.stack(
            [cm.state_dict()[k].float() for cm in client_models], 0
        ).mean(0)
    global_model.load_state_dict(global_dict)
    return global_model


# ============================================================================
# 主流程
# ============================================================================

def run_ggeur_fedavg(args):
    """运行GGEUR FedAvg"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"设备: {device}")

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # ========== Step 1: 加载数据 ==========
    logger.info("="*60)
    logger.info("Step 1: 加载论文预计算数据")
    logger.info("="*60)

    # 加载训练特征
    client_features = {}
    for domain in DOMAINS:
        client_id = DOMAIN_TO_CLIENT[domain]
        features_by_class = load_train_features(domain)
        client_features[client_id] = features_by_class

        total_samples = sum(len(f) for f in features_by_class.values())
        logger.info(f"  {domain} (Client {client_id}): {total_samples} 训练样本, "
                   f"{len(features_by_class)} 个类别")

    # 加载测试特征
    test_features_all = []
    test_labels_all = []
    for domain in DOMAINS:
        feats, labels = load_test_features(domain)
        if feats is not None:
            test_features_all.append(feats)
            test_labels_all.append(labels)
            logger.info(f"  {domain}: {len(feats)} 测试样本")

    test_features = np.vstack(test_features_all)
    test_labels = np.concatenate(test_labels_all)
    logger.info(f"  测试集总计: {len(test_features)} 样本")

    # 加载协方差矩阵
    cov_matrices = load_covariance_matrices()
    logger.info(f"  加载 {len(cov_matrices)} 个协方差矩阵")

    # ========== Step 2: 计算原型 ==========
    logger.info("\n" + "="*60)
    logger.info("Step 2: 计算原型")
    logger.info("="*60)

    prototypes = {}
    for client_id, features_by_class in client_features.items():
        prototypes[client_id] = {}
        for class_id, features in features_by_class.items():
            if len(features) > 0:
                prototypes[client_id][class_id] = np.mean(features, axis=0)

    # ========== Step 3: 离线数据增强 ==========
    logger.info("\n" + "="*60)
    logger.info("Step 3: 离线数据增强")
    logger.info("="*60)

    augmented_data = {}
    client_ids = list(client_features.keys())

    for client_id in client_ids:
        all_features = []
        all_labels = []

        # 其他域
        other_client_ids = [cid for cid in client_ids if cid != client_id]

        for class_id in range(NUM_CLASSES):
            # 原始特征
            original = client_features[client_id].get(class_id, np.empty((0, 512)))
            if len(original.shape) == 1 or len(original) == 0:
                original = np.empty((0, 512))

            # 协方差
            cov = cov_matrices.get(class_id, np.zeros((512, 512)))

            # 其他域原型
            other_protos = []
            for other_id in other_client_ids:
                if class_id in prototypes[other_id]:
                    other_protos.append(prototypes[other_id][class_id])

            # 跳过空协方差
            if np.all(cov == 0):
                if len(original) > 0:
                    all_features.append(original)
                    all_labels.append(np.full(len(original), class_id, dtype=np.int64))
                continue

            # 增强
            augmented = augment_class_features(
                original, cov, other_protos,
                target_size=args.target_size,
                n_per_sample=args.n_per_sample
            )

            if len(augmented) > 0:
                all_features.append(augmented)
                all_labels.append(np.full(len(augmented), class_id, dtype=np.int64))

        if len(all_features) > 0:
            augmented_data[client_id] = (
                np.vstack(all_features),
                np.concatenate(all_labels)
            )
            logger.info(f"  Client {client_id}: {len(augmented_data[client_id][0])} 增强样本")

    # ========== Step 4: 联邦训练 ==========
    logger.info("\n" + "="*60)
    logger.info("Step 4: 联邦训练 (FedAvg)")
    logger.info("="*60)
    logger.info(f"  轮数: {args.num_rounds}")
    logger.info(f"  本地epochs: {args.local_epochs}")
    logger.info(f"  学习率: {args.lr}")
    logger.info(f"  批量大小: {args.batch_size}")

    # 创建模型
    global_model = MyNet(NUM_CLASSES).to(device)
    client_models = [MyNet(NUM_CLASSES).to(device) for _ in client_ids]

    criterion = nn.CrossEntropyLoss()

    # 创建数据加载器
    client_loaders = []
    for client_id in client_ids:
        features, labels = augmented_data[client_id]
        dataset = FeatureDataset(features, labels)
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
        client_loaders.append(loader)

    test_dataset = FeatureDataset(test_features, test_labels)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # 训练循环
    best_acc = 0.0

    for round_idx in range(args.num_rounds):
        # 分发全局模型
        for client_model in client_models:
            client_model.load_state_dict(global_model.state_dict())

        # 本地训练
        for client_idx, loader in enumerate(client_loaders):
            client_model = client_models[client_idx]
            optimizer = optim.Adam(client_model.parameters(), lr=args.lr)

            for epoch in range(args.local_epochs):
                train_loss, train_acc = train_epoch(
                    client_model, loader, optimizer, criterion, device
                )

        # FedAvg聚合
        global_model = federated_averaging(global_model, client_models)

        # 评估
        test_acc = evaluate(global_model, test_loader, device)
        if test_acc > best_acc:
            best_acc = test_acc

        if (round_idx + 1) % 10 == 0 or round_idx == 0:
            logger.info(f"  Round {round_idx + 1}/{args.num_rounds}: "
                       f"Test Acc = {test_acc:.4f} ({test_acc*100:.2f}%), "
                       f"Best = {best_acc:.4f}")

    # ========== 最终结果 ==========
    logger.info("\n" + "="*60)
    logger.info("最终结果")
    logger.info("="*60)
    logger.info(f"  最佳测试精度: {best_acc:.4f} ({best_acc*100:.2f}%)")

    # 每域精度
    logger.info("\n  每域精度:")
    for domain in DOMAINS:
        feats, labels = load_test_features(domain)
        if feats is not None:
            domain_dataset = FeatureDataset(feats, labels)
            domain_loader = DataLoader(domain_dataset, batch_size=args.batch_size, shuffle=False)
            domain_acc = evaluate(global_model, domain_loader, device)
            logger.info(f"    {domain}: {domain_acc:.4f} ({domain_acc*100:.2f}%)")

    return best_acc


def main():
    parser = argparse.ArgumentParser(description='GGEUR_Clip FederatedScope运行脚本')

    # 数据路径
    parser.add_argument('--data_root', type=str, default=PAPER_DATA_ROOT,
                       help='论文数据根目录')

    # 增强参数
    parser.add_argument('--target_size', type=int, default=50,
                       help='每类目标样本数')
    parser.add_argument('--n_per_sample', type=int, default=50,
                       help='每个原始样本生成的样本数')

    # 训练参数 - 论文默认值
    parser.add_argument('--num_rounds', type=int, default=50,
                       help='联邦轮数')
    parser.add_argument('--local_epochs', type=int, default=1,
                       help='本地训练epochs')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='批量大小')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='学习率')
    parser.add_argument('--seed', type=int, default=12345,
                       help='随机种子')

    args = parser.parse_args()

    run_ggeur_fedavg(args)


if __name__ == '__main__':
    main()
