from federatedscope.core.data import BaseDataTranslator
from federatedscope.register import register_data
from torch.utils.data import Dataset, Subset
from torchvision import transforms
from sklearn.model_selection import train_test_split
import numpy as np
import os
from PIL import Image


class AIDDataset(Dataset):
    """基础数据集类（无任何预处理）"""
    def __init__(self, root_dir):
        self.image_paths = []
        self.labels = []
        self.label_to_idx = {}

        # 过滤非目录项并建立类别索引
        classes = sorted([
            cls for cls in os.listdir(root_dir)
            if os.path.isdir(os.path.join(root_dir, cls))
        ])

        for idx, cls in enumerate(classes):
            self.label_to_idx[cls] = idx
            cls_dir = os.path.join(root_dir, cls)
            valid_extensions = ('.jpg', '.jpeg', '.png')
            img_files = [
                f for f in os.listdir(cls_dir)
                if f.lower().endswith(valid_extensions)
            ]

            self.image_paths.extend(
                [os.path.join(cls_dir, f) for f in img_files])
            self.labels.extend([idx] * len(img_files))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # 返回原始未处理的数据
        return Image.open(
            self.image_paths[idx]).convert('RGB'), self.labels[idx]


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


def apply_split_transform(full_dataset, idx_list, transform):
    """安全拆分并应用变换的三合一操作"""
    subset = Subset(full_dataset, idx_list)
    return TransformWrapper(subset, transform=transform)


def load_my_data(config, client_cfgs=None):
    # 标准化参数（与预训练模型匹配）
    pretrained_mean = [0.5, 0.5, 0.5]
    pretrained_std = [0.5, 0.5, 0.5]

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

    # 加载原始未处理的数据集
    full_dataset = AIDDataset("/home/lzy/OpenDataLab___AID/data/AID")

    # 执行分层拆分（保证每类分布一致）
    indices = np.arange(len(full_dataset))
    labels = [full_dataset[i][1] for i in indices]

    # 第一次拆分：训练+暂存集 与 测试集
    train_tmp_idx, test_idx = train_test_split(indices,
                                               test_size=0.2,
                                               stratify=labels,
                                               random_state=config.seed)

    # 第二次拆分：从训练+暂存集中拆分验证集
    train_idx, val_idx = train_test_split(
        train_tmp_idx,
        test_size=0.25,  # 0.2 total = 0.8*0.25=0.2
        stratify=[labels[i] for i in train_tmp_idx],
        random_state=config.seed)

    # 应用不同变换到各子集（互不干扰）
    train_set = apply_split_transform(full_dataset, train_idx, train_transform)
    val_set = apply_split_transform(full_dataset, val_idx, eval_transform)
    test_set = apply_split_transform(full_dataset, test_idx, eval_transform)

    # 组装框架需要的三元组格式
    split_datasets = (train_set, val_set, test_set)

    # 传递给Translator处理联邦划分
    translator = BaseDataTranslator(config, client_cfgs)
    fs_data = translator(split_datasets)

    # 调试信息（可选）
    # if config.verbose >= 1:
    #     sample, _ = train_set[0]
    #     print(f"[验证] 训练样本尺寸: {sample.shape}")
    #     print(f"[验证] 拆分统计: 训练={len(train_idx)},
    #           验证={len(val_idx)}, 测试={len(test_idx)}")

    return fs_data, config


def call_my_data(config, client_cfgs=None):
    if config.data.type == "AID":
        return load_my_data(config, client_cfgs=client_cfgs)


register_data("AID", call_my_data)
