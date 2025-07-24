import os
import numpy as np
import random
from PIL import Image

import torch
from torch.utils.data import Subset, Dataset
from torchvision import transforms
import os
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import logging
from federatedscope.contrib.data.utils import dirichlet_split, get_label_distribution
from federatedscope.contrib.data.utils import (
    adaptive_cluster_clients,
    save_cluster_results,
    sample_data,
)
from federatedscope.core.data.base_data import ClientData, StandaloneDataDict

logger = logging.getLogger(__name__)

#  domainnet 数据集
# domains = ["clipart", "infograph", "painting", "quickdraw", "real", "sketch"]


# 取出所有域的共有类别
def load_common_classes(base_path, domains):
    domain_classes = []

    # 收集每个域的类别集合
    for domain in domains:
        domain_path = os.path.join(base_path, domain)
        if not os.path.exists(domain_path):
            logger.warning(f"Domain {domain} does not exist at {domain_path}")
            continue
        # 获取该域的所有类别（目录名）
        classes = set(os.listdir(domain_path))
        domain_classes.append(classes)

    # 计算所有域类别的交集
    if domain_classes:
        common_classes = set.intersection(*domain_classes)
        logger.info(f"Common classes across domains: {common_classes}")
        return common_classes
    else:
        logger.error("No valid domains found")
        return set()


# 随机选择指定数量的类
def random_select_classes(base_path, num_classes):
    domains = ["clipart", "infograph", "painting", "quickdraw", "real", "sketch"]
    all_classes = load_common_classes(base_path, domains)
    if num_classes > len(all_classes):
        return list(all_classes)
    selected_classes = random.sample(all_classes, num_classes)
    return selected_classes


class DomainNetDataset(Dataset):
    def __init__(self, root_dir, domain, split, class_list, transform=None):
        """
        Args:
            root_dir (string): 数据集根目录
            domain (string): 域名称（如'clipart', 'painting'等）
            split (string): 'train' 或 'test'
            class_list (list): 要加载的类别名称列表（如['zebra', 'dog']）
            transform (callable): 应用于图像的变换
        """
        self.root_dir = root_dir
        self.domain = domain
        self.split = split
        self.transform = transform
        self.class_list = class_list
        logger.info(
            f"Loading {split} data for domain '{domain}' with classes: {class_list}"
        )
        # 读取划分文件
        split_file = os.path.join(root_dir, f"{domain}_{split}.txt")
        if not os.path.exists(split_file):
            raise FileNotFoundError(f"划分文件不存在: {split_file}")

        # 解析文件内容：路径和类别
        with open(split_file, "r") as f:
            lines = f.readlines()

        # 提取路径和标签
        self.data = []
        self.class_to_idx = {}
        self.idx_to_class = {}

        valid_classes = class_list
        # 创建类别映射
        sorted_classes = sorted(valid_classes)
        self.class_to_idx = {
            cls_name: idx for idx, cls_name in enumerate(sorted_classes)
        }
        self.idx_to_class = {
            idx: cls_name for idx, cls_name in enumerate(sorted_classes)
        }

        # 第二遍：收集符合条件的数据
        for line in lines:
            path, _ = line.strip().split(" ")
            class_name = os.path.basename(os.path.dirname(path))

            if class_name in valid_classes:
                img_path = os.path.join(root_dir, path)
                label = self.class_to_idx[class_name]
                self.data.append((img_path, label))

        if not self.data:
            raise ValueError(f"在域 {domain} 的{split}集中未找到指定类别的数据")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, label = self.data[idx]

        try:
            image = Image.open(img_path).convert("RGB")
        except:
            logger.error(f"无法打开图像: {img_path}")
            return None

        if self.transform:
            image = self.transform(image)

        return image, label


def load_domainnet_data(config, client_cfgs=None):
    data_root = config.data.root
    alpha = config.data.splitter_args[0]["alpha"]
    total_clients = config.federate.client_num
    classes = config.model.out_channels
    sampled_classes = random_select_classes(data_root, classes)

    transform_train = transforms.Compose(
        [
            transforms.Resize([256, 256]),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation((-30, 30)),
            transforms.ToTensor(),
        ]
    )
    transform_test = transforms.Compose(
        [
            transforms.Resize([256, 256]),
            transforms.ToTensor(),
        ]
    )

    train_ds = []
    test_ds = []
    # 加载数据集
    for domain in ["clipart", "painting", "quickdraw", "infograph", "sketch"]:
        train_ds.append(
            DomainNetDataset(
                root_dir=data_root,
                domain=domain,
                split="train",
                class_list=sampled_classes,
                transform=transform_train,
            )
        )
        test_ds.append(
            DomainNetDataset(
                root_dir=data_root,
                domain=domain,
                split="test",
                class_list=sampled_classes,
                transform=transform_test,
            )
        )

        # train_samples_num.append(len(train_ds[-1]))

    train_ds = [ds if len(ds) <= 10000 else sample_data(ds, 10000) for ds in train_ds]
    test_ds = [ds if len(ds) <= 2000 else sample_data(ds, 2000) for ds in test_ds]
    logger.info(f"Loaded {len(train_ds)} domains with {len(test_ds)} test datasets.")
    train_samples_num = []
    for ds in train_ds:
        train_samples_num.append(len(ds))
    logger.info(f"Train samples per domain: {train_samples_num}")
    # 划分训练集
    train_splits = []
    # total_samples = sum(train_samples_num)
    client_for_each_domain = [int(0.2 * total_clients)] * 5
    # 确保总数正确
    for ds, num_clients in zip(train_ds, client_for_each_domain):
        train_idxs = dirichlet_split(ds, num_clients, alpha=alpha)
        train_splits.extend([Subset(ds, idxs) for idxs in train_idxs])
    logger.info(
        f"Created {len(train_splits)} training splits across {len(train_ds)} domains with {total_clients} clients."
    )
    # 划分测试集
    test_splits = []
    for ds, num_clients in zip(test_ds, client_for_each_domain):
        test_idxs = dirichlet_split(ds, num_clients, alpha=alpha)
        test_splits.extend([Subset(ds, idxs) for idxs in test_idxs])
    logger.info(
        f"Created {len(test_splits)} test splits across {len(test_ds)} domains with {total_clients} clients."
    )
    # 获取标签分布
    train_label_dist = get_label_distribution(
        train_splits, num_classes=len(sampled_classes)
    )
    test_label_dist = get_label_distribution(
        test_splits, num_classes=len(sampled_classes)
    )

    # 按照分布聚类客户端
    peer_communities = adaptive_cluster_clients(train_label_dist)
    logger.info(f"Peer communities: {peer_communities}")
    save_cluster_results(
        os.path.join(data_root, "peer_communities.json"),
        train_label_dist,
        test_label_dist,
        peer_communities,
    )

    client_data_dict = {}
    for cid, (tr, te) in enumerate(zip(train_splits, test_splits), start=1):
        client_data_dict[cid] = ClientData(
            client_cfg=config, train=tr, val=None, test=te
        )
    fs_data = StandaloneDataDict(client_data_dict, config)
    return fs_data, config


def call_domainnet_data(config, client_cfgs=None):
    """
    Load DomainNet data for FederatedScope.
    """
    if config.data.type == "domainnet":
        return load_domainnet_data(config, client_cfgs)


if __name__ == "__main__":
    DATA_ROOT = "data/domainnet"
    # all_classes = load_all_classes(DATA_ROOT)
    sampled_classes = random_select_classes(DATA_ROOT, 100)

    transforms = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    clipart_train = DomainNetDataset(
        root_dir=DATA_ROOT,
        domain="clipart",
        split="train",
        class_list=sampled_classes,
        transform=transforms,
    )
    print(
        f"Loaded {len(clipart_train)} training samples from 'clipart' domain with classes: {sampled_classes}"
    )

    clipart_test = DomainNetDataset(
        root_dir=DATA_ROOT,
        domain="clipart",
        split="test",
        class_list=sampled_classes,
        transform=transforms,
    )
    print(
        f"Loaded {len(clipart_test)} test samples from 'clipart' domain with classes: {sampled_classes}"
    )

    clipart_train_idxs = dirichlet_split(clipart_train, num_clients=10, alpha=0.5)
    train_splits = [Subset(clipart_train, idxs) for idxs in clipart_train_idxs]
    clipart_train_label_dist = get_label_distribution(
        train_splits, num_classes=len(sampled_classes)
    )

    clipart_train_dataloaders_0 = torch.utils.data.DataLoader(
        train_splits[0],
        batch_size=32,
        shuffle=True,
        num_workers=4,
    )

else:
    # 如果不是直接运行此脚本，则注册数据加载函数
    from federatedscope.register import register_data

    register_data("domainnet", call_domainnet_data)
