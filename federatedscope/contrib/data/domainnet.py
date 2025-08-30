from federatedscope.core.data import BaseDataTranslator
from federatedscope.register import register_data
from torch.utils.data import Dataset, Subset
from torchvision import transforms
from sklearn.model_selection import train_test_split
import numpy as np
import os
from PIL import Image


class DomainnetDataset(Dataset):
    """DomainNet 数据集读取器，支持多个域，每个域下按类组织。

    结构假设：
    root_dir/
      <domain_A>/
        <class_1>/*.jpg
        <class_2>/*.jpg
        ...
      <domain_B>/
      <domain_C>/
        ...

    输出仅为 (image, label)，但对象内部保留 `labels`、
    `image_paths`、`domain_ids` 便于 splitter 使用。
    """
    def __init__(self, root_dir: str):
        self.root_dir = root_dir
        self.image_paths = []
        self.labels = []  # 全局类别 id（域之间共享一个 label 空间）
        self.domain_ids = []  # 每个样本所属域 id [0..num_domains-1]
        self.class_to_idx = {}
        self.idx_to_class = []
        self.domain_names = []

        # 枚举域
        domains = [
            d for d in os.listdir(root_dir)
            if os.path.isdir(os.path.join(root_dir, d))
        ]
        domains.sort()
        self.domain_names = domains

        # 收集所有类名（跨域去重，确保共享 label 空间）
        class_names_set = set()
        for d in domains:
            dpath = os.path.join(root_dir, d)
            for cls in os.listdir(dpath):
                if os.path.isdir(os.path.join(dpath, cls)):
                    class_names_set.add(cls)
        self.idx_to_class = sorted(class_names_set)
        self.class_to_idx = {c: i for i, c in enumerate(self.idx_to_class)}

        # 逐域逐类收集样本
        valid_ext = ('.jpg', '.jpeg', '.png', '.bmp')
        for dom_id, d in enumerate(domains):
            dpath = os.path.join(root_dir, d)
            for cls in os.listdir(dpath):
                cdir = os.path.join(dpath, cls)
                if not os.path.isdir(cdir):
                    continue
                if cls not in self.class_to_idx:
                    # 理论上不会发生，已全局收集
                    continue
                cls_idx = self.class_to_idx[cls]
                files = [
                    f for f in os.listdir(cdir)
                    if f.lower().endswith(valid_ext)
                ]
                for f in files:
                    self.image_paths.append(os.path.join(cdir, f))
                    self.labels.append(cls_idx)
                    self.domain_ids.append(dom_id)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert('RGB')
        label = self.labels[idx]
        return img, label


class TransformWrapper(Dataset):
    """对任意 Dataset/Subset 应用变换，不改变取样协议。

    注意：不改变 __getitem__ 的返回，只做图像增强与标准化。
    """
    def __init__(self, dataset: Dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        if self.transform:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.dataset)

    def __getattr__(self, name):
        # 允许外部访问底层数据集的属性（如 labels/domain_ids ）
        if name in ('dataset', 'transform'):
            return super().__getattribute__(name)
        return getattr(self.dataset, name)


def _apply_split_transform(full_dataset: Dataset, idx_list, transform):
    subset = Subset(full_dataset, idx_list)
    return TransformWrapper(subset, transform=transform)


def load_domainnet_data(config, client_cfgs=None):
    # 归一化参数（常见 ImageNet 风格，若需可按需调整）
    mean = [0.5, 0.5, 0.5]
    std = [0.5, 0.5, 0.5]

    train_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    eval_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    # 加载完整数据集
    root = "/root/domainnet"
    full_dataset = DomainnetDataset(root)

    # 按 60/20/20 进行分层拆分（train/val/test）
    indices = np.arange(len(full_dataset))
    labels = [full_dataset.labels[i] for i in indices]

    # 先划分出 20% 测试集
    train_tmp_idx, test_idx = train_test_split(indices,
                                               test_size=0.2,
                                               stratify=labels,
                                               random_state=config.seed)
    # 再从剩余 80% 中划分出 25% 作为验证集（即总量的 20%）
    train_idx, val_idx = train_test_split(
        train_tmp_idx,
        test_size=0.25,
        stratify=[labels[i] for i in train_tmp_idx],
        random_state=config.seed)

    train_set = _apply_split_transform(full_dataset, train_idx,
                                       train_transform)
    val_set = _apply_split_transform(full_dataset, val_idx, eval_transform)
    test_set = _apply_split_transform(full_dataset, test_idx, eval_transform)

    split_datasets = (train_set, val_set, test_set)

    translator = BaseDataTranslator(config, client_cfgs)
    fs_data = translator(split_datasets)
    return fs_data, config


def call_domainnet_data(config, client_cfgs=None):
    if config.data.type == "Domainnet":
        return load_domainnet_data(config, client_cfgs=client_cfgs)


register_data("Domainnet", call_domainnet_data)
