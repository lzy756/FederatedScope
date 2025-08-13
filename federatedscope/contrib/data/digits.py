import os
import numpy as np
from PIL import Image
from sklearn.datasets import load_svmlight_file
import scipy.io as sio
import torch
from torch.utils.data import Dataset, TensorDataset, Subset, ConcatDataset
from torchvision import datasets, transforms
from torchvision.datasets import VisionDataset
from federatedscope.core.data import BaseDataTranslator
import json
import logging
import random
from sklearn.cluster import DBSCAN
from scipy.stats import entropy
from federatedscope.register import register_data
from federatedscope.core.data import ClientData, StandaloneDataDict
from federatedscope.contrib.splitter.fedgs_splitter import FedGS_Splitter
logger = logging.getLogger(__name__)

##########################
# 聚类相关函数（参考cross_domain_digits.py）
##########################

def get_label_distribution(subsets: list, num_classes=10):
    """
    计算每个子集的标签分布
    :param subsets: list of Subset objects or datasets
    :param num_classes: number of classes in the dataset
    :return: list of distributions for each subset
    """
    distributions = []
    for subset in subsets:
        labels = []
        # 获取标签
        if hasattr(subset, 'dataset'):
            # 如果是Subset对象
            for idx in subset.indices:
                _, label = subset.dataset[idx]
                labels.append(label)
        else:
            # 如果是普通dataset
            for i in range(len(subset)):
                _, label = subset[i]
                labels.append(label)

        labels = np.array(labels)
        dist = np.zeros(num_classes, dtype=int)
        for label in labels:
            dist[label] += 1
        distributions.append(dist)
    return distributions

##########################
# 数据采样函数
##########################

def sample_dataset(dataset, fraction):
    """
    从数据集中随机采样指定比例的数据

    Args:
        dataset: 原始数据集
        fraction: 采样比例 (0, 1]

    Returns:
        Subset: 采样后的数据集子集
    """
    if fraction >= 1.0:
        return dataset

    total_size = len(dataset)
    sample_size = int(total_size * fraction)

    # 生成随机索引
    indices = list(range(total_size))
    random.shuffle(indices)
    sampled_indices = indices[:sample_size]

    logger.info(f"从 {total_size} 个样本中采样 {sample_size} 个样本 (比例: {fraction:.3f})")

    return Subset(dataset, sampled_indices)

##########################
# 数据加载部分（示例）
##########################

def load_usps(data_root, transform):
    usps_dir = os.path.join(data_root, 'USPS')
    X_tr_sp, y_tr = load_svmlight_file(os.path.join(usps_dir, 'usps'), n_features=256)
    X_te_sp, y_te = load_svmlight_file(os.path.join(usps_dir, 'usps.t'), n_features=256)
    X_tr = X_tr_sp.toarray().reshape(-1, 16, 16).astype('uint8')
    X_te = X_te_sp.toarray().reshape(-1, 16, 16).astype('uint8')
    y_tr = y_tr.astype(int) - 1
    y_te = y_te.astype(int) - 1

    class USPSDataset(TensorDataset):
        def __init__(self, X, y, transform):
            self.X, self.y, self.transform = X, y, transform
        def __len__(self):
            return len(self.y)
        def __getitem__(self, idx):
            img = Image.fromarray(self.X[idx], mode='L')
            img = self.transform(img)
            return img, int(self.y[idx])

    return USPSDataset(X_tr, y_tr, transform), USPSDataset(X_te, y_te, transform)

def load_svhn(data_root, transform):
    def load_mat_svhn(split):
        mat = sio.loadmat(os.path.join(data_root, f"SVHN/{split}_32x32.mat"))
        X = np.transpose(mat['X'], (3, 0, 1, 2))
        y = mat['y'].reshape(-1).astype(int)
        y[y == 10] = 0
        class SVHNDataset(Dataset):
            def __init__(self, X, y, transform):
                self.X, self.y, self.transform = X, y, transform
            def __len__(self):
                return len(self.y)
            def __getitem__(self, idx):
                img = Image.fromarray(self.X[idx])
                img = self.transform(img)
                return img, int(self.y[idx])
        return SVHNDataset(X, y, transform)
    return load_mat_svhn('train'), load_mat_svhn('test')

def load_syn(data_root, transform):
    from torchvision.datasets import ImageFolder
    syn_root = os.path.join(data_root, 'SYN')
    train_ds = ImageFolder(root=os.path.join(syn_root, 'imgs_train'), transform=transform)
    test_ds = ImageFolder(root=os.path.join(syn_root, 'imgs_valid'), transform=transform)
    return train_ds, test_ds

class MNISTM(VisionDataset):
    """MNIST-M Dataset for FederatedScope.

    This dataset loads MNIST-M data from pre-downloaded .pt files.
    """

    training_file = "mnist_m_train.pt"
    test_file = "mnist_m_test.pt"
    classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

    def __init__(self, root, train=True, transform=None, target_transform=None):
        """Initialize MNIST-M dataset.

        Args:
            root (str): Root directory where data is stored
            train (bool): If True, load training data; otherwise load test data
            transform: Transform to apply to images
            target_transform: Transform to apply to targets
        """
        super(MNISTM, self).__init__(root, transform=transform, target_transform=target_transform)

        self.train = train

        # Check if data files exist
        if not self._check_exists():
            raise RuntimeError(
                f"Dataset not found in {self.data_folder}. "
                f"Please ensure {self.training_file} and {self.test_file} are present."
            )

        # Load data
        if self.train:
            data_file = self.training_file
        else:
            data_file = self.test_file

        data_path = os.path.join(self.data_folder, data_file)
        logger.info(f"Loading MNIST-M data from: {data_path}")

        self.data, self.targets = torch.load(data_path)

        # Convert targets to long tensor if needed
        if not isinstance(self.targets, torch.LongTensor):
            self.targets = self.targets.long()

        logger.info(f"Loaded MNIST-M {'train' if train else 'test'} data: "
                   f"{len(self.data)} samples, {len(torch.unique(self.targets))} classes")

    def __getitem__(self, index):
        """Get item by index.

        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class
        """
        img, target = self.data[index], int(self.targets[index])

        # Convert tensor to PIL Image
        # MNIST-M images are RGB, so we need to handle the tensor format properly
        if isinstance(img, torch.Tensor):
            # If tensor is in CHW format, convert to HWC for PIL
            if img.dim() == 3 and img.shape[0] == 3:
                img = img.permute(1, 2, 0)
            img = img.numpy()

        # Ensure the image is in the right format for PIL
        if img.dtype != 'uint8':
            img = (img * 255).astype('uint8')

        img = Image.fromarray(img, mode="RGB")

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        """Return size of dataset."""
        return len(self.data)

    @property
    def data_folder(self):
        """Return the folder containing the data files."""
        return os.path.join(self.root, 'mnist_m')

    @property
    def class_to_idx(self):
        """Return mapping from class names to indices."""
        return {cls: i for i, cls in enumerate(self.classes)}

    def _check_exists(self):
        """Check if data files exist."""
        train_path = os.path.join(self.data_folder, self.training_file)
        test_path = os.path.join(self.data_folder, self.test_file)
        return os.path.exists(train_path) and os.path.exists(test_path)

    def extra_repr(self):
        """Return extra representation string."""
        return f"Split: {'Train' if self.train else 'Test'}"

def _merge_data_dictionaries(config, data_dicts):
    """
    Merge multiple federated data dictionaries into a single one

    Args:
        config: Original configuration object
        data_dicts: List of federated data dictionaries to merge

    Returns:
        StandaloneDataDict: Merged federated data dictionary
    """
    from torch.utils.data import ConcatDataset

    # Calculate total clients dynamically
    total_clients = sum(len(data_dict) - 1 for data_dict in data_dicts)  # -1 to exclude server

    # Create merged config
    tmp_config_merged = config.clone()
    tmp_config_merged.merge_from_list(['federate.client_num', total_clients])

    # Create merged data dictionary
    merged_data_dict = {}

    # Merge server data (ID=0)
    server_train = ConcatDataset([data_dict[0].train_data for data_dict in data_dicts])
    server_val = ConcatDataset([data_dict[0].val_data for data_dict in data_dicts])
    server_test = ConcatDataset([data_dict[0].test_data for data_dict in data_dicts])
    merged_data_dict[0] = ClientData(tmp_config_merged, train=server_train, val=server_val, test=server_test)

    # Add clients from each data dictionary with proper ID offset
    current_client_id = 1
    for data_dict in data_dicts:
        num_clients_in_dict = len(data_dict) - 1  # Exclude server (ID=0)
        for client_id in range(1, num_clients_in_dict + 1):
            merged_data_dict[current_client_id] = data_dict[client_id]
            # Update client configuration to merged config
            merged_data_dict[current_client_id].setup(tmp_config_merged)
            current_client_id += 1

    # Create final StandaloneDataDict
    return StandaloneDataDict(merged_data_dict, tmp_config_merged)

def _perform_clustering_analysis(data_dict, data_root):
    """
    Perform clustering analysis on federated data using FedGS_Splitter

    Args:
        data_dict: Federated data dictionary
        data_root: Root directory for saving results
    """
    try:
        # Collect training and test subsets
        train_subsets = []
        test_subsets = []

        # Extract data from each client (skip server id=0)
        for client_id in range(1, len(data_dict)):
            client_data = data_dict[client_id]

            # Access original datasets rather than DataLoaders
            if hasattr(client_data, 'train_data') and client_data.train_data is not None:
                train_subsets.append(client_data.train_data)
            if hasattr(client_data, 'test_data') and client_data.test_data is not None:
                test_subsets.append(client_data.test_data)

        logger.info(f"收集到 {len(train_subsets)} 个训练子集, {len(test_subsets)} 个测试子集")

        if train_subsets and test_subsets:
            # Calculate label distributions using existing function
            logger.info("计算客户端标签分布...")
            train_distributions = get_label_distribution(train_subsets, num_classes=10)
            test_distributions = get_label_distribution(test_subsets, num_classes=10)

            # Create FedGS_Splitter instance for clustering
            logger.info("使用FedGS_Splitter执行客户端聚类...")
            num_clients = len(train_distributions)
            
            # Initialize splitter with appropriate parameters
            splitter = FedGS_Splitter(
                client_num=num_clients,
                alpha=0.5,
                distance_threshold=0.5,
                min_groups=2,
                max_groups=min(10, num_clients // 2),  # 动态调整最大组数
                balance_weight=2.0,
                balance_tolerance=0.2
            )
            
            # Store distributions in splitter
            splitter.data_info = [{"distribution": np.array(dist)} for dist in train_distributions]
            splitter.test_data_info = [{"distribution": np.array(dist)} for dist in test_distributions]
            
            # Perform clustering using FedGS_Splitter
            splitter.build_peer_communities(train_distributions)
            
            # Save clustering results using FedGS_Splitter's method
            cluster_file = os.path.join(data_root, "peer_communities.json")
            splitter.save_cluster_results(cluster_file)
            
            logger.info(f"聚类结果: {splitter.peer_communities}")
            logger.info(f"聚类结果已保存到: {cluster_file}")

        else:
            logger.warning("无法获取客户端数据子集，跳过聚类步骤")

    except Exception as e:
        logger.error(f"聚类过程中出现错误: {e}")
        logger.info("继续返回联邦数据，但没有聚类信息")

##########################
# 客户端数量分配函数
##########################

def calculate_client_allocation(dataset_sizes, total_clients=100, min_clients_per_domain=5):
    """
    根据数据集大小比例分配客户端数量
    
    Args:
        dataset_sizes: 各数据集的样本数量列表
        total_clients: 总客户端数量
        min_clients_per_domain: 每个域的最小客户端数量
    
    Returns:
        list: 每个数据集分配的客户端数量
    """
    total_samples = sum(dataset_sizes)
    
    # 计算基于比例的初始分配
    initial_allocation = []
    for size in dataset_sizes:
        proportion = size / total_samples
        allocated = max(min_clients_per_domain, int(proportion * total_clients))
        initial_allocation.append(allocated)
    
    # 调整总数使其等于目标总客户端数
    current_total = sum(initial_allocation)
    if current_total != total_clients:
        # 按比例调整最大的几个域
        diff = total_clients - current_total
        largest_domains = sorted(range(len(initial_allocation)), 
                               key=lambda i: initial_allocation[i], reverse=True)
        
        for i in range(abs(diff)):
            domain_idx = largest_domains[i % len(largest_domains)]
            if diff > 0:
                initial_allocation[domain_idx] += 1
            elif initial_allocation[domain_idx] > min_clients_per_domain:
                initial_allocation[domain_idx] -= 1
    
    return initial_allocation

##########################
# 主入口函数，注册给FederatedScope调用
##########################

from federatedscope.core.data.base_data import StandaloneDataDict, ClientData
from federatedscope.register import register_data

def load_digits_data(config, client_cfgs=None):
    data_root = config.data.root

    # 设置随机种子以确保可重现性
    random.seed(42)
    np.random.seed(42)

    # 加载数据集
    transform = transforms.Compose(
        [
            transforms.Resize((32, 32)),  # 统一缩放到32x32
            # 将灰度图复制为3通道（如果是RGB则无变化）
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),  # 转为Tensor并归一化到[0,1]
            transforms.Normalize(
                mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]
            ),  # 标准化到[-1,1]
        ]
    )

    # 先加载所有数据集以获取样本数量
    logger.info("Loading all datasets to calculate sample sizes...")
    
    # MNIST数据集
    mnist_train = MNISTM(root=data_root, train=True, transform=transform)
    mnist_test = MNISTM(root=data_root, train=False, transform=transform)
    logger.info(f"Original MNISTM dataset: train={len(mnist_train)}, test={len(mnist_test)}")
    mnist_train = sample_dataset(mnist_train, 1/6)
    mnist_test = sample_dataset(mnist_test, 1/6)
    logger.info(f"Sampled MNIST-M dataset: train={len(mnist_train)}, test={len(mnist_test)}")

    # USPS数据集
    usps_train, usps_test = load_usps(data_root, transform)
    logger.info(f"Loaded USPS dataset: train={len(usps_train)}, test={len(usps_test)}")

    # SVHN数据集
    svhn_train, svhn_test = load_svhn(data_root, transform)
    logger.info(f"Original SVHN dataset: train={len(svhn_train)}, test={len(svhn_test)}")
    svhn_train = sample_dataset(svhn_train, 1/7)
    svhn_test = sample_dataset(svhn_test, 1/7)
    logger.info(f"Sampled SVHN dataset: train={len(svhn_train)}, test={len(svhn_test)}")

    # SYN数据集
    syn_train, syn_test = load_syn(data_root, transform)
    logger.info(f"Loaded SYN dataset: train={len(syn_train)}, test={len(syn_test)}")

    # 合并四个域的测试集，并随机采样四分之一作为统一测试集
    combined_test = ConcatDataset([mnist_test, usps_test, svhn_test, syn_test])
    combined_total = len(combined_test)
    combined_sample_size = max(1, int(combined_total * 0.25))
    combined_indices = list(range(combined_total))
    random.shuffle(combined_indices)
    combined_sampled_test = Subset(combined_test, combined_indices[:combined_sample_size])
    logger.info(
        f"Combined test set size={combined_total}, sampled 1/4 for translators' test size={combined_sample_size}")

    # 计算训练集样本数量用于客户端分配
    dataset_train_sizes = [len(mnist_train), len(usps_train), len(svhn_train), len(syn_train)]
    dataset_names = ["MNIST-M", "USPS", "SVHN", "SYN"]
    
    # 根据样本数量比例分配客户端数量
    total_clients = getattr(config.federate, 'client_num', 100)  # 从配置获取总客户端数，默认100
    client_allocation = calculate_client_allocation(dataset_train_sizes, total_clients)
    
    logger.info("Client allocation based on dataset sizes:")
    for i, (name, size, clients) in enumerate(zip(dataset_names, dataset_train_sizes, client_allocation)):
        percentage = (size / sum(dataset_train_sizes)) * 100
        logger.info(f"{name}: {size} samples ({percentage:.1f}%) -> {clients} clients")
    
    logger.info(f"Total clients allocated: {sum(client_allocation)}")

    # 使用分配的客户端数量创建联邦数据集
    # MNIST数据集
    mnist_split_datasets = (mnist_train, mnist_test, combined_sampled_test)
    tmp_config = config.clone()
    tmp_config.merge_from_list(['federate.client_num', client_allocation[0]])
    translator = BaseDataTranslator(tmp_config, client_cfgs)
    mnist_data_dict = translator(mnist_split_datasets)
    logger.info(f"Successfully created federated MNIST dataset with {len(mnist_data_dict) - 1} clients")

    # USPS数据集
    usps_split_datasets = (usps_train, usps_test, combined_sampled_test)
    tmp_config = config.clone()
    tmp_config.merge_from_list(['federate.client_num', client_allocation[1]])
    translator = BaseDataTranslator(tmp_config, client_cfgs)
    usps_data_dict = translator(usps_split_datasets)
    logger.info(f"Successfully created federated USPS dataset with {len(usps_data_dict) - 1} clients")

    # SVHN数据集
    svhn_split_datasets = (svhn_train, svhn_test, combined_sampled_test)
    tmp_config = config.clone()
    tmp_config.merge_from_list(['federate.client_num', client_allocation[2]])
    translator = BaseDataTranslator(tmp_config, client_cfgs)
    svhn_data_dict = translator(svhn_split_datasets)
    logger.info(f"Successfully created federated SVHN dataset with {len(svhn_data_dict) - 1} clients")
    
    # SYN数据集
    syn_split_datasets = (syn_train, syn_test, combined_sampled_test)
    tmp_config = config.clone()
    tmp_config.merge_from_list(['federate.client_num', client_allocation[3]])
    translator = BaseDataTranslator(tmp_config, client_cfgs)
    syn_data_dict = translator(syn_split_datasets)
    logger.info(f"Successfully created federated SYN dataset with {len(syn_data_dict) - 1} clients")

    data_dicts = [mnist_data_dict, usps_data_dict, svhn_data_dict, syn_data_dict]
    # Merge all data dictionaries
    merged_data_dict = _merge_data_dictionaries(config, data_dicts)

    # Perform clustering analysis
    _perform_clustering_analysis(merged_data_dict, data_root)

    return merged_data_dict, config

def call_digits_data(config, client_cfgs=None):
    if config.data.type == "digits":
        return load_digits_data(config, client_cfgs)

from federatedscope.register import register_data
register_data("digits", call_digits_data)