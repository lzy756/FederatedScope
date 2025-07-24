import os
import numpy as np
from PIL import Image
from sklearn.datasets import load_svmlight_file
import scipy.io as sio
from torch.utils.data import Subset, Dataset, TensorDataset
from torchvision import datasets, transforms
import logging
from federatedscope.register import register_data
from federatedscope.core.data.base_data import ClientData, StandaloneDataDict

from federatedscope.contrib.data.utils import (
    dirichlet_split,
    uniform_split,
    get_label_distribution,
    adaptive_cluster_clients,
    save_cluster_results,
    sample_data,
)

logger = logging.getLogger(__name__)


def load_usps(data_root, transform):
    usps_dir = os.path.join(data_root, "USPS")
    X_tr_sp, y_tr = load_svmlight_file(os.path.join(usps_dir, "usps"), n_features=256)
    X_te_sp, y_te = load_svmlight_file(os.path.join(usps_dir, "usps.t"), n_features=256)
    X_tr = X_tr_sp.toarray().reshape(-1, 16, 16).astype("uint8")
    X_te = X_te_sp.toarray().reshape(-1, 16, 16).astype("uint8")
    y_tr = y_tr.astype(int) - 1
    y_te = y_te.astype(int) - 1

    class USPSDataset(TensorDataset):
        def __init__(self, X, y, transform):
            self.X, self.y, self.transform = X, y, transform

        def __len__(self):
            return len(self.y)

        def __getitem__(self, idx):
            img = Image.fromarray(self.X[idx], mode="L")
            img = self.transform(img)
            return img, int(self.y[idx])

    return USPSDataset(X_tr, y_tr, transform), USPSDataset(X_te, y_te, transform)


def load_svhn(data_root, transform):
    def load_mat_svhn(split):
        mat = sio.loadmat(os.path.join(data_root, f"SVHN/{split}_32x32.mat"))
        X = np.transpose(mat["X"], (3, 0, 1, 2))
        y = mat["y"].reshape(-1).astype(int)
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

    return load_mat_svhn("train"), load_mat_svhn("test")


def load_syn(data_root, transform):
    from torchvision.datasets import ImageFolder

    syn_root = os.path.join(data_root, "SYN")
    train_ds = ImageFolder(
        root=os.path.join(syn_root, "imgs_train"), transform=transform
    )
    test_ds = ImageFolder(
        root=os.path.join(syn_root, "imgs_valid"), transform=transform
    )
    return train_ds, test_ds


def load_cross_domain_data(config, client_cfgs=None):
    data_root = config.data.root
    alpha = config.data.splitter_args[0].get("alpha", 0.5)
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

    mnist_train = datasets.MNIST(
        root=data_root, train=True, transform=transform, download=True
    )
    mnist_test = datasets.MNIST(
        root=data_root, train=False, transform=transform, download=True
    )

    usps_train, usps_test = load_usps(data_root, transform)
    svhn_train, svhn_test = load_svhn(data_root, transform)
    syn_train, syn_test = load_syn(data_root, transform)

    mnist_train = sample_data(mnist_train, 6000)
    svhn_train = sample_data(svhn_train, 10000)
    mnist_test = sample_data(mnist_test, 2000)
    svhn_test = sample_data(svhn_test, 2000)

    logger.info(
        f"Loaded datasets: MNIST, USPS, SVHN, SYN with {len(mnist_train)}, {len(usps_train)}, {len(svhn_train)}, {len(syn_train)} "
        f"train samples and {len(mnist_test)}, {len(usps_test)}, {len(svhn_test)}, {len(syn_test)} test samples each."
    )

    total_clients = config.federate.get("client_num", 100)  # 默认100个客户端

    client_for_mnist = int(0.1 * total_clients)
    client_for_usps = int(0.1 * total_clients)
    client_for_svhn = int(0.4 * total_clients)
    client_for_syn = total_clients - (
        client_for_mnist + client_for_usps + client_for_svhn
    )
    test_splits = []
    mnist_test_idxs = uniform_split(mnist_test, client_for_mnist)
    usps_test_idxs = uniform_split(usps_test, client_for_usps)
    svhn_test_idxs = uniform_split(svhn_test, client_for_svhn)
    syn_test_idxs = uniform_split(syn_test, client_for_syn)
    test_splits.extend([Subset(mnist_test, idxs) for idxs in mnist_test_idxs])
    test_splits.extend([Subset(usps_test, idxs) for idxs in usps_test_idxs])
    test_splits.extend([Subset(svhn_test, idxs) for idxs in svhn_test_idxs])
    test_splits.extend([Subset(syn_test, idxs) for idxs in syn_test_idxs])

    logger.info(
        f"分配客户端数量: MNIST: {client_for_mnist}, USPS: {client_for_usps}, SVHN: {client_for_svhn}, SYN: {client_for_syn}"
    )
    # 训练集按照不同的域名进行划分
    mnist_train_idxs = dirichlet_split(mnist_train, client_for_mnist, alpha=alpha)
    usps_train_idxs = dirichlet_split(usps_train, client_for_usps, alpha=alpha)
    svhn_train_idxs = dirichlet_split(svhn_train, client_for_svhn, alpha=alpha)
    syn_train_idxs = dirichlet_split(syn_train, client_for_syn, alpha=alpha)

    train_splits = []
    train_splits.extend([Subset(mnist_train, idxs) for idxs in mnist_train_idxs])
    train_splits.extend([Subset(usps_train, idxs) for idxs in usps_train_idxs])
    train_splits.extend([Subset(svhn_train, idxs) for idxs in svhn_train_idxs])
    train_splits.extend([Subset(syn_train, idxs) for idxs in syn_train_idxs])

    # 计算分布
    train_distributions = get_label_distribution(train_splits, num_classes=10)
    test_distributions = get_label_distribution(test_splits, num_classes=10)

    # 聚类划分
    peer_communities = adaptive_cluster_clients(train_distributions)

    logger.info(f"Peer communities: {peer_communities}")
    # 保存聚类结果
    save_cluster_results(
        os.path.join(data_root, f"peer_communities.json"),
        train_distributions,
        test_distributions,
        peer_communities,
    )

    # 构建 FederatedScope 格式数据
    client_data_dict = {}

    for cid, (tr, te) in enumerate(zip(train_splits, test_splits), start=1):
        client_data_dict[cid] = ClientData(
            client_cfg=config, train=tr, val=None, test=te
        )
    fs_data = StandaloneDataDict(client_data_dict, config)
    return fs_data, config


def call_cross_domain_data(config, client_cfgs=None):
    if config.data.type == "cross_domain":
        return load_cross_domain_data(config, client_cfgs)


if __name__ == "__main__":
    from federatedscope.core.configs.config import global_cfg

    # 设置数据根目录
    global_cfg.data.root = "./data"
    # 设置数据类型
    global_cfg.data.type = "cross_domain"
    # 调用数据加载函数
    fs_data, config = load_cross_domain_data(global_cfg)
else:
    # 如果不是直接运行此脚本，则注册数据加载函数
    from federatedscope.register import register_data

    register_data("cross_domain", call_cross_domain_data)
