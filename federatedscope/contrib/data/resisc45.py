import os
import logging
from typing import Optional, Tuple, Any

import torch
from torch.utils.data import Dataset, Subset
from torchvision import transforms
from PIL import Image
import numpy as np

from federatedscope.core.data import BaseDataTranslator
from federatedscope.register import register_data

logger = logging.getLogger(__name__)
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"


class RESISC45Dataset(Dataset):
    """基础RESISC45数据集类（无任何预处理）"""
    def __init__(self, data_list):
        """
        Args:
            data_list: List of samples from huggingface datasets
        """
        self.data_list = data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        sample = self.data_list[idx]

        # Extract image and label - return PIL Image directly
        image = sample['image'] if isinstance(
            sample['image'], Image.Image) else Image.fromarray(sample['image'])
        label = sample['label']

        return image, label


class TransformWrapper(Dataset):
    """数据集变换包装器，支持每个子集独立变换"""
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        if self.transform:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.dataset)


def load_resisc45_data(config, client_cfgs=None):
    """
    Load RESISC45 dataset from Hugging Face datasets

    Args:
        config: Configuration object containing dataset parameters
        client_cfgs: Client-specific configurations

    Returns:
        FederatedScope data format and updated config
    """
    try:
        import datasets
        from datasets import load_dataset
    except ImportError:
        raise ImportError(
            "Please install datasets library: pip install datasets")

    logger.info("Loading RESISC45 dataset from Hugging Face")

    # Load dataset from Hugging Face
    try:
        dataset = load_dataset("timm/resisc45", trust_remote_code=True)
    except Exception as e:
        logger.error(f"Failed to load timm/resisc45 dataset: {e}")
        raise

    # 标准化参数（与预训练模型匹配）
    pretrained_mean = [0.485, 0.456, 0.406]  # ImageNet标准
    pretrained_std = [0.229, 0.224, 0.225]

    # 定义变换策略
    train_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(pretrained_mean, pretrained_std)
    ])

    eval_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(pretrained_mean, pretrained_std)
    ])

    # 使用数据集自带的分割（train/validation/test）
    if ('train' not in dataset or 'validation' not in dataset
            or 'test' not in dataset):
        raise ValueError(
            "Dataset should contain 'train', 'validation', and 'test' splits")

    # 转换为我们的数据格式
    train_data_list = list(dataset['train'])
    val_data_list = list(dataset['validation'])
    test_data_list = list(dataset['test'])

    logger.info(
        f"Loaded train: {len(train_data_list)}, val: {len(val_data_list)}, "
        f"test: {len(test_data_list)} samples from HuggingFace")

    # 创建各分割的数据集（无变换）
    train_dataset = RESISC45Dataset(train_data_list)
    val_dataset = RESISC45Dataset(val_data_list)
    test_dataset = RESISC45Dataset(test_data_list)

    # 应用不同变换到各子集
    train_set = TransformWrapper(train_dataset, transform=train_transform)
    val_set = TransformWrapper(val_dataset, transform=eval_transform)
    test_set = TransformWrapper(test_dataset, transform=eval_transform)

    # 组装框架需要的三元组格式
    split_datasets = (train_set, val_set, test_set)

    # 传递给Translator处理联邦划分
    translator = BaseDataTranslator(config, client_cfgs)
    fs_data = translator(split_datasets)

    # Update config if provided
    if config is not None:
        # Set number of classes (RESISC45 has 45 classes)
        if hasattr(config, 'model') and hasattr(config.model, 'out_channels'):
            config.model.out_channels = 45

        # Log dataset info
        logger.info("RESISC45 dataset loaded successfully")
        logger.info(f"拆分统计: 训练={len(train_data_list)}, "
                    f"验证={len(val_data_list)}, 测试={len(test_data_list)}")
        logger.info("Number of classes: 45")

    return fs_data, config


def call_resisc45_data(config, client_cfgs=None):
    """
    Entry point for FederatedScope data loading system
    """
    if config.data.type in ["resisc45", "timm/resisc45"]:
        return load_resisc45_data(config, client_cfgs=client_cfgs)


# Register the data loader with FederatedScope
register_data("resisc45", call_resisc45_data)
register_data("timm/resisc45", call_resisc45_data)

# Example usage and configuration helper
# def create_resisc45_config():
#     """
#     Create a sample configuration for RESISC45 dataset
#     """
#     from federatedscope.core.configs.config import CN

#     config = CN()

#     # Basic configuration
#     config.use_gpu = True
#     config.device = 0
#     config.expname_tag = 'RESISC45_ViT_fedsak_lda'
#     config.seed = 12345

#     # Early stopping
#     config.early_stop = CN()
#     config.early_stop.patience = 0

#     # Federated learning configuration
#     config.federate = CN()
#     config.federate.client_num = 20
#     config.federate.method = 'fedsak'
#     config.federate.mode = 'standalone'
#     config.federate.total_round_num = 50
#     config.federate.sample_client_rate = 1.0

#     # Data configuration
#     config.data = CN()
#     config.data.type = 'resisc45'
#     config.data.splitter = 'RESISC45splitter'

#     # Dataloader configuration
#     config.dataloader = CN()
#     config.dataloader.batch_size = 32

#     # Model configuration for RESISC45 (45 classes)
#     config.model = CN()
#     config.model.type = 'ViT'
#     config.model.out_channels = 45

#     # Aggregator configuration
#     config.aggregator = CN()
#     config.aggregator.lambda_ = 2
#     config.aggregator.lr_shared = 0.05
#     config.aggregator.filter_keys = [
#         'embeddings.position',
#         'encoder.layer.9',
#         'encoder.layer.10',
#         'encoder.layer.11',
#         'classifier'
#     ]

#     # Trainer configuration
#     config.trainer = CN()
#     config.trainer.type = 'fedsak'

#     # FedSAK specific configuration
#     config.fedsak = CN()
#     config.fedsak.share_patterns = [
#         'embeddings.position',
#         'encoder.layer.9',
#         'encoder.layer.10',
#         'encoder.layer.11',
#         'classifier'
#     ]

#     # Training configuration
#     config.train = CN()
#     config.train.local_update_steps = 1
#     config.train.batch_or_epoch = 'epoch'
#     config.train.optimizer = CN()
#     config.train.optimizer.type = 'Adam'
#     config.train.optimizer.lr = 2e-4
#     config.train.optimizer.weight_decay = 0.01

#     # Gradient configuration
#     config.grad = CN()
#     config.grad.grad_clip = 5.0

#     # Criterion configuration
#     config.criterion = CN()
#     config.criterion.type = 'CrossEntropyLoss'

#     # Evaluation configuration
#     config.eval = CN()
#     config.eval.freq = 2
#     config.eval.metrics = ['acc', 'correct']
#     config.eval.count_flops = False

#     return config

if __name__ == "__main__":
    import os
    from collections import defaultdict
    from PIL import Image as PILImage
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError(
            "Please install datasets library: pip install datasets")

    # 创建输出文件夹
    output_dir = "/home/lzy/resisc45"
    os.makedirs(output_dir, exist_ok=True)

    print("Loading RESISC45 dataset from Hugging Face...")

    # 直接加载数据集
    dataset = load_dataset("timm/resisc45", trust_remote_code=True)

    # 使用训练集来提取样本
    train_data = list(dataset['train'])

    # 按类别组织数据
    class_samples = defaultdict(list)
    class_names = {}

    for sample in train_data:
        label = sample['label']
        image = sample['image']
        filename = sample.get('filename',
                              f'sample_{len(class_samples[label])}')

        # 如果图片是numpy数组，转换为PIL图像
        if not isinstance(image, PILImage.Image):
            image = PILImage.fromarray(image)

        class_samples[label].append((image, filename))

        # 获取类别名称（从文件名推断）
        if label not in class_names and 'filename' in sample:
            class_name = sample['filename'].split(
                '_')[0] if '_' in sample['filename'] else f'class_{label}'
            class_names[label] = class_name

    # 为没有文件名的类别设置默认名称
    for label in class_samples:
        if label not in class_names:
            class_names[label] = f'class_{label}'

    print(f"Found {len(class_samples)} classes in the dataset")

    # 为每个类别创建拼接图像（最多10张图片）
    for label, samples in class_samples.items():
        class_name = class_names[label]
        print(f"Processing class {label}: {class_name} "
              f"(total samples: {len(samples)})")

        # 取前10个样本
        selected_samples = samples[:10]

        if not selected_samples:
            continue

        # 创建拼接图像
        n_samples = len(selected_samples)

        # 计算网格大小（尽量接近正方形）
        cols = min(5, n_samples)  # 最多5列
        rows = (n_samples + cols - 1) // cols

        # 获取图像尺寸（假设所有图像尺寸相同）
        sample_img = selected_samples[0][0]
        img_width, img_height = sample_img.size

        # 创建拼接画布
        canvas_width = cols * img_width
        canvas_height = rows * img_height

        # 创建matplotlib图像
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
        if rows == 1 and cols == 1:
            axes = [axes]
        elif rows == 1 or cols == 1:
            axes = axes.flatten()
        else:
            axes = axes.flatten()

        # 添加图像到网格中
        for i, (img, filename) in enumerate(selected_samples):
            row = i // cols
            col = i % cols
            ax_idx = i

            if ax_idx < len(axes):
                axes[ax_idx].imshow(img)
                axes[ax_idx].set_title(filename, fontsize=8)
                axes[ax_idx].axis('off')

        # 隐藏多余的子图
        for i in range(len(selected_samples), len(axes)):
            axes[i].axis('off')

        # 设置整体标题
        fig.suptitle(
            f'Class {label}: {class_name} '
            f'(showing {len(selected_samples)} samples)',
            fontsize=14,
            fontweight='bold')

        plt.tight_layout()

        # 保存图像
        output_path = os.path.join(output_dir, f'class_{label:02d}.png')
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"Saved: {output_path}")

    print(f"\nAll class visualization images saved to: {output_dir}")
    print(f"Total classes processed: {len(class_samples)}")

    # 另外创建一个总览图，显示所有类别的第一张图片
    print("Creating overview image with one sample from each class...")

    # 计算总览图的网格大小
    total_classes = len(class_samples)
    overview_cols = min(9, total_classes)  # 最多9列
    overview_rows = (total_classes + overview_cols - 1) // overview_cols

    fig, axes = plt.subplots(overview_rows,
                             overview_cols,
                             figsize=(overview_cols * 2, overview_rows * 2))
    if overview_rows == 1 and overview_cols == 1:
        axes = [axes]
    elif overview_rows == 1 or overview_cols == 1:
        axes = axes.flatten()
    else:
        axes = axes.flatten()

    # 添加每个类别的第一张图片
    sorted_labels = sorted(class_samples.keys())
    for i, label in enumerate(sorted_labels):
        if i < len(axes) and class_samples[label]:
            first_img = class_samples[label][0][0]
            class_name = class_names[label]

            axes[i].imshow(first_img)
            axes[i].set_title(f'{label}: {class_name}', fontsize=8)
            axes[i].axis('off')

    # 隐藏多余的子图
    for i in range(len(sorted_labels), len(axes)):
        axes[i].axis('off')

    fig.suptitle('RESISC45 Dataset - All 45 Classes Overview',
                 fontsize=16,
                 fontweight='bold')
    plt.tight_layout()

    # 保存总览图
    overview_path = os.path.join(output_dir, 'resisc45_overview.png')
    plt.savefig(overview_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Overview image saved to: {overview_path}")
    print("Dataset visualization complete!")
