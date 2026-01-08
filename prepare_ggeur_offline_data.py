#!/usr/bin/env python
"""
GGEUR离线数据预处理脚本

从原始图像数据集生成GGEUR所需的离线数据：
1. CLIP特征提取 (ViT-B-32)
2. LDS数据划分
3. 协方差矩阵计算与聚合
4. 保存预处理结果

用法:
    python prepare_ggeur_offline_data.py \
        --data_root /path/to/OfficeHomeDataset \
        --output_dir ./ggeur_preprocessed \
        --clip_model ViT-B-32
"""

import os
import sys
import argparse
import logging
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
from typing import Dict, List, Tuple

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# CLIP特征提取
# ============================================================================

class CLIPFeatureExtractor:
    """CLIP特征提取器"""

    def __init__(self, model_name='ViT-B-32', device='cuda', weights_path=None):
        """
        Args:
            model_name: CLIP模型名称 (论文使用ViT-B-32)
            device: 设备
            weights_path: 本地模型权重路径 (可选)
        """
        self.device = device
        self.model_name = model_name

        logger.info(f"加载CLIP模型: {model_name}")
        if weights_path:
            logger.info(f"使用本地权重: {weights_path}")

        try:
            import open_clip

            if weights_path and os.path.exists(weights_path):
                # 使用本地权重
                self.model, _, self.preprocess = open_clip.create_model_and_transforms(
                    model_name, pretrained=weights_path
                )
                logger.info(f"从本地加载权重: {weights_path}")
            else:
                # 从网络下载
                self.model, _, self.preprocess = open_clip.create_model_and_transforms(
                    model_name, pretrained='openai'
                )
            self.use_open_clip = True
        except ImportError:
            try:
                import clip
                self.model, self.preprocess = clip.load(model_name, device=device)
                self.use_open_clip = False
            except ImportError:
                raise ImportError("请安装 open_clip 或 clip: pip install open-clip-torch")

        self.model.eval().to(device)
        logger.info(f"CLIP模型加载完成，使用 {'open_clip' if self.use_open_clip else 'clip'}")

    @torch.no_grad()
    def extract_features(self, dataloader) -> Tuple[np.ndarray, np.ndarray]:
        """
        提取数据集的CLIP特征

        Args:
            dataloader: 数据加载器

        Returns:
            features: [N, 512] numpy数组
            labels: [N] numpy数组
        """
        all_features = []
        all_labels = []

        for images, labels in tqdm(dataloader, desc="提取CLIP特征"):
            images = images.to(self.device)

            if self.use_open_clip:
                features = self.model.encode_image(images)
            else:
                features = self.model.encode_image(images)

            # 归一化 (CLIP特征通常需要归一化)
            features = features / features.norm(dim=-1, keepdim=True)

            all_features.append(features.cpu().numpy())
            all_labels.append(labels.numpy())

        return np.vstack(all_features), np.concatenate(all_labels)


# ============================================================================
# 论文核心算法
# ============================================================================

def compute_covariance(features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """计算均值和协方差"""
    mean = np.mean(features, axis=0)
    centered = features - mean
    cov = (1 / len(features)) * centered.T @ centered
    return mean, cov


def aggregate_covariances(client_data: Dict[int, Dict[int, np.ndarray]],
                         num_classes: int) -> Dict[int, np.ndarray]:
    """
    聚合所有客户端的协方差矩阵 - 完全匹配论文

    Args:
        client_data: {client_id: {class_id: features}}
        num_classes: 类别数

    Returns:
        global_covs: {class_id: covariance_matrix}
    """
    global_covs = {}

    for class_id in range(num_classes):
        sample_counts = []
        means = []
        covs = []

        for client_id, client_features in client_data.items():
            if class_id not in client_features:
                continue
            features = client_features[class_id]
            if len(features) == 0:
                continue

            mean, cov = compute_covariance(features)
            sample_counts.append(len(features))
            means.append(mean)
            covs.append(cov)

        if not sample_counts:
            global_covs[class_id] = np.zeros((512, 512))
            continue

        total = sum(sample_counts)
        combined_mean = sum(n * m for n, m in zip(sample_counts, means)) / total
        combined_cov = sum(n * c for n, c in zip(sample_counts, covs)) / total
        combined_cov += sum(n * np.outer(m - combined_mean, m - combined_mean)
                          for n, m in zip(sample_counts, means)) / total

        global_covs[class_id] = combined_cov

    return global_covs


def apply_lds(features_by_class: Dict[int, np.ndarray],
              alpha: float, domain_idx: int, num_domains: int,
              seed: int = 12345) -> Dict[int, np.ndarray]:
    """
    应用Label Distribution Skew (LDS)

    Args:
        features_by_class: {class_id: features}
        alpha: Dirichlet参数
        domain_idx: 当前域索引
        num_domains: 域总数
        seed: 随机种子

    Returns:
        lds_features: {class_id: subsampled_features}
    """
    np.random.seed(seed)
    num_classes = max(features_by_class.keys()) + 1

    # 生成Dirichlet分布
    dirichlet = np.random.dirichlet([alpha] * num_domains, num_classes)  # [num_classes, num_domains]

    lds_features = {}
    for class_id, features in features_by_class.items():
        if len(features) == 0:
            lds_features[class_id] = features
            continue

        # 该客户端在该类别的比例
        proportion = dirichlet[class_id, domain_idx]
        n_samples = int(len(features) * proportion * num_domains)
        n_samples = min(n_samples, len(features))

        if n_samples > 0:
            indices = np.random.choice(len(features), n_samples, replace=False)
            lds_features[class_id] = features[indices]
        else:
            lds_features[class_id] = np.empty((0, 512))

    return lds_features


# ============================================================================
# 主预处理流程
# ============================================================================

def preprocess_office_home(args):
    """预处理Office-Home数据集"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"设备: {device}")

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    domains = ['Art', 'Clipart', 'Product', 'Real_World']
    num_classes = 65

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    train_dir = os.path.join(args.output_dir, 'train_features')
    test_dir = os.path.join(args.output_dir, 'test_features')
    cov_dir = os.path.join(args.output_dir, 'cov_matrices')
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    os.makedirs(cov_dir, exist_ok=True)

    # 初始化CLIP
    clip_extractor = CLIPFeatureExtractor(args.clip_model, device, args.clip_weights)

    # 数据变换
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                           std=[0.26862954, 0.26130258, 0.27577711])
    ])

    # ========== Step 1: 提取CLIP特征 ==========
    logger.info("="*60)
    logger.info("Step 1: 提取CLIP特征")
    logger.info("="*60)

    all_train_features = {}  # {domain_idx: {class_id: features}}
    all_test_features = {}   # {domain_idx: (features, labels)}

    for domain_idx, domain in enumerate(domains):
        domain_path = os.path.join(args.data_root, domain)
        if not os.path.exists(domain_path):
            logger.warning(f"域目录不存在: {domain_path}")
            continue

        logger.info(f"\n处理域: {domain}")

        # 加载数据集
        dataset = datasets.ImageFolder(domain_path, transform=transform)
        logger.info(f"  总样本数: {len(dataset)}")

        # 划分训练/测试 (80/20)
        indices = np.random.permutation(len(dataset))
        train_size = int(0.8 * len(dataset))
        train_indices = indices[:train_size]
        test_indices = indices[train_size:]

        # 训练集特征提取
        train_subset = torch.utils.data.Subset(dataset, train_indices)
        train_loader = DataLoader(train_subset, batch_size=args.batch_size,
                                 shuffle=False, num_workers=args.num_workers)
        train_features, train_labels = clip_extractor.extract_features(train_loader)
        logger.info(f"  训练特征: {train_features.shape}")

        # 测试集特征提取
        test_subset = torch.utils.data.Subset(dataset, test_indices)
        test_loader = DataLoader(test_subset, batch_size=args.batch_size,
                                shuffle=False, num_workers=args.num_workers)
        test_features, test_labels = clip_extractor.extract_features(test_loader)
        logger.info(f"  测试特征: {test_features.shape}")

        # 按类别组织训练特征
        features_by_class = {}
        for class_id in range(num_classes):
            mask = train_labels == class_id
            if mask.sum() > 0:
                features_by_class[class_id] = train_features[mask]

        # 应用LDS
        if args.apply_lds:
            logger.info(f"  应用LDS (alpha={args.lds_alpha})")
            features_by_class = apply_lds(
                features_by_class, args.lds_alpha,
                domain_idx, len(domains), args.seed
            )
            total_after_lds = sum(len(f) for f in features_by_class.values())
            logger.info(f"  LDS后样本数: {total_after_lds}")

        all_train_features[domain_idx] = features_by_class
        all_test_features[domain_idx] = (test_features, test_labels)

        # 保存训练特征
        domain_train_dir = os.path.join(train_dir, domain)
        os.makedirs(domain_train_dir, exist_ok=True)
        for class_id, features in features_by_class.items():
            np.save(os.path.join(domain_train_dir,
                                f'client_{domain_idx}_class_{class_id}_original_features.npy'),
                   features)
            np.save(os.path.join(domain_train_dir,
                                f'client_{domain_idx}_class_{class_id}_labels.npy'),
                   np.full(len(features), class_id, dtype=np.int64))

        # 保存测试特征
        domain_test_dir = os.path.join(test_dir, domain)
        os.makedirs(domain_test_dir, exist_ok=True)
        np.save(os.path.join(domain_test_dir, f'{domain}_test_features.npy'), test_features)
        np.save(os.path.join(domain_test_dir, f'{domain}_test_labels.npy'), test_labels)

    # ========== Step 2: 计算并聚合协方差矩阵 ==========
    logger.info("\n" + "="*60)
    logger.info("Step 2: 计算并聚合协方差矩阵")
    logger.info("="*60)

    global_covs = aggregate_covariances(all_train_features, num_classes)

    for class_id, cov in global_covs.items():
        np.save(os.path.join(cov_dir, f'class_{class_id}_cov_matrix.npy'), cov)

    logger.info(f"保存 {len(global_covs)} 个协方差矩阵")

    # ========== Step 3: 保存元数据 ==========
    logger.info("\n" + "="*60)
    logger.info("Step 3: 保存元数据")
    logger.info("="*60)

    metadata = {
        'domains': domains,
        'num_classes': num_classes,
        'clip_model': args.clip_model,
        'lds_alpha': args.lds_alpha if args.apply_lds else None,
        'seed': args.seed,
    }

    import json
    with open(os.path.join(args.output_dir, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"\n预处理完成！数据保存到: {args.output_dir}")
    logger.info("\n目录结构:")
    logger.info(f"  {args.output_dir}/")
    logger.info(f"  ├── train_features/     # 训练特征")
    logger.info(f"  │   ├── Art/")
    logger.info(f"  │   ├── Clipart/")
    logger.info(f"  │   ├── Product/")
    logger.info(f"  │   └── Real_World/")
    logger.info(f"  ├── test_features/      # 测试特征")
    logger.info(f"  ├── cov_matrices/       # 协方差矩阵")
    logger.info(f"  └── metadata.json       # 元数据")


def preprocess_pacs(args):
    """预处理PACS数据集"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"设备: {device}")

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    domains = ['art_painting', 'cartoon', 'photo', 'sketch']
    num_classes = 7

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    train_dir = os.path.join(args.output_dir, 'train_features')
    test_dir = os.path.join(args.output_dir, 'test_features')
    cov_dir = os.path.join(args.output_dir, 'cov_matrices')
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    os.makedirs(cov_dir, exist_ok=True)

    # 初始化CLIP
    clip_extractor = CLIPFeatureExtractor(args.clip_model, device, args.clip_weights)

    # 数据变换
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                           std=[0.26862954, 0.26130258, 0.27577711])
    ])

    logger.info("="*60)
    logger.info("预处理PACS数据集")
    logger.info("="*60)

    all_train_features = {}
    all_test_features = {}

    for domain_idx, domain in enumerate(domains):
        domain_path = os.path.join(args.data_root, domain)
        if not os.path.exists(domain_path):
            logger.warning(f"域目录不存在: {domain_path}")
            continue

        logger.info(f"\n处理域: {domain}")

        dataset = datasets.ImageFolder(domain_path, transform=transform)
        logger.info(f"  总样本数: {len(dataset)}")

        # 划分训练/测试
        indices = np.random.permutation(len(dataset))
        train_size = int(0.8 * len(dataset))
        train_indices = indices[:train_size]
        test_indices = indices[train_size:]

        # 提取特征
        train_subset = torch.utils.data.Subset(dataset, train_indices)
        train_loader = DataLoader(train_subset, batch_size=args.batch_size,
                                 shuffle=False, num_workers=args.num_workers)
        train_features, train_labels = clip_extractor.extract_features(train_loader)

        test_subset = torch.utils.data.Subset(dataset, test_indices)
        test_loader = DataLoader(test_subset, batch_size=args.batch_size,
                                shuffle=False, num_workers=args.num_workers)
        test_features, test_labels = clip_extractor.extract_features(test_loader)

        # 按类别组织
        features_by_class = {}
        for class_id in range(num_classes):
            mask = train_labels == class_id
            if mask.sum() > 0:
                features_by_class[class_id] = train_features[mask]

        # 应用LDS
        if args.apply_lds:
            features_by_class = apply_lds(
                features_by_class, args.lds_alpha,
                domain_idx, len(domains), args.seed
            )

        all_train_features[domain_idx] = features_by_class
        all_test_features[domain_idx] = (test_features, test_labels)

        # 保存
        domain_train_dir = os.path.join(train_dir, domain)
        os.makedirs(domain_train_dir, exist_ok=True)
        for class_id, features in features_by_class.items():
            np.save(os.path.join(domain_train_dir,
                                f'client_{domain_idx}_class_{class_id}_original_features.npy'),
                   features)

        domain_test_dir = os.path.join(test_dir, domain)
        os.makedirs(domain_test_dir, exist_ok=True)
        np.save(os.path.join(domain_test_dir, f'{domain}_test_features.npy'), test_features)
        np.save(os.path.join(domain_test_dir, f'{domain}_test_labels.npy'), test_labels)

    # 聚合协方差
    global_covs = aggregate_covariances(all_train_features, num_classes)
    for class_id, cov in global_covs.items():
        np.save(os.path.join(cov_dir, f'class_{class_id}_cov_matrix.npy'), cov)

    logger.info(f"\n预处理完成！数据保存到: {args.output_dir}")


def main():
    parser = argparse.ArgumentParser(description='GGEUR离线数据预处理')

    # 数据集
    parser.add_argument('--data_root', type=str, required=True,
                       help='原始数据集根目录')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='输出目录')
    parser.add_argument('--dataset', type=str, default='office_home',
                       choices=['office_home', 'pacs'],
                       help='数据集类型')

    # CLIP
    parser.add_argument('--clip_model', type=str, default='ViT-B-32',
                       help='CLIP模型 (默认: ViT-B-32)')
    parser.add_argument('--clip_weights', type=str, default=None,
                       help='本地CLIP模型权重路径 (如不指定则从网络下载)')

    # LDS
    parser.add_argument('--apply_lds', action='store_true',
                       help='是否应用LDS')
    parser.add_argument('--lds_alpha', type=float, default=0.1,
                       help='LDS Dirichlet参数 (默认: 0.1)')

    # 其他
    parser.add_argument('--batch_size', type=int, default=32,
                       help='批量大小')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='数据加载线程数')
    parser.add_argument('--seed', type=int, default=12345,
                       help='随机种子')

    args = parser.parse_args()

    if args.dataset == 'office_home':
        preprocess_office_home(args)
    elif args.dataset == 'pacs':
        preprocess_pacs(args)


if __name__ == '__main__':
    main()
