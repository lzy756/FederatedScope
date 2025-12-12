import os
import pickle

import torch
from torch.utils.data import TensorDataset
from torchvision.datasets import MNIST

from federatedscope.register import register_data
from federatedscope.core.data import StandaloneDataDict, ClientData


def _ensure_channel_first(tensor):
    if tensor.ndim != 4:
        raise ValueError(
            f"MNIST-M tensor should be 4-D, but got shape {tensor.shape}.")
    # Handle numpy arrays
    if isinstance(tensor, torch.Tensor):
        arr = tensor
    else:
        arr = torch.tensor(tensor)
    if arr.shape[1] == 3:
        return arr.float()
    if arr.shape[-1] == 3:
        return arr.permute(0, 3, 1, 2).float()
    raise ValueError(
        f"Cannot infer channel axis for MNIST-M tensor with shape "
        f"{tensor.shape}.")


def _load_mnist(root):
    train = MNIST(root=root, train=True, download=True)
    test = MNIST(root=root, train=False, download=True)

    train_images = train.data[:55000].float()
    train_images = (train_images > 0).float() * 255.
    train_images = train_images.unsqueeze(1).repeat(1, 3, 1, 1)
    train_labels = train.targets[:55000].long()

    test_images = (test.data > 0).float() * 255.
    test_images = test_images.unsqueeze(1).repeat(1, 3, 1, 1)
    test_labels = test.targets.long()
    return train_images, train_labels, test_images, test_labels


def _load_mnistm(root):
    pt_path = os.path.join(root, 'mnistm_cache.pt')
    pkl_path = os.path.join(root, 'mnistm_data.pkl')
    if os.path.exists(pt_path):
        mnistm = torch.load(pt_path)
    elif os.path.exists(pkl_path):
        with open(pkl_path, 'rb') as f:
            mnistm = pickle.load(f)
    else:
        raise FileNotFoundError(
            f"Cannot find MNIST-M cache at {pt_path} or {pkl_path}. Please "
            f"place the converted dataset under `cfg.data.root`.")

    train = _ensure_channel_first(mnistm['train'])
    test = _ensure_channel_first(mnistm['test'])
    return train, test


def _build_client_data(cfg, source_data, target_data):
    client = ClientData(cfg, train=None, val=None, test=None)
    client.fedmm_data = {'source': source_data, 'target': target_data}
    sample_x, sample_y = _sample_placeholder(source_data, target_data)
    client['train'] = {
        'x': sample_x,
        'y': sample_y,
    }
    return client


def _sample_placeholder(source_data, target_data):
    for data in [source_data, target_data]:
        if data[0].numel() > 0:
            x = data[0][:1].detach().cpu().numpy()
            y = data[1][:1].detach().cpu().numpy()
            return x, y
    x = torch.zeros(1, 3, 28, 28).numpy()
    y = torch.zeros(1, dtype=torch.long).numpy()
    return x, y


def call_fedmm_mnistm(config, client_cfgs):
    if config.data.type.lower() != 'fedmm_mnistm':
        return None

    if config.federate.client_num != 2:
        raise ValueError("FedMM currently expects exactly 2 clients.")

    root = config.data.root
    mnist_train_x, mnist_train_y, mnist_test_x, mnist_test_y = _load_mnist(
        root)
    mnistm_train_x, mnistm_test_x = _load_mnistm(root)

    # MNIST-M labels follow the original MNIST labels
    mnistm_train_y = mnist_train_y.clone()
    mnistm_test_y = mnist_test_y.clone()

    num_source = mnist_train_x.size(0)
    num_target = mnistm_train_x.size(0)

    split_source = int(num_source * config.fedmm.source_ratio)
    split_target = int(num_target * config.fedmm.target_ratio)

    client1_source = (
        mnist_train_x[:split_source].clone(),
        mnist_train_y[:split_source].clone())
    client2_source = (
        mnist_train_x[split_source:].clone(),
        mnist_train_y[split_source:].clone())

    client1_target = (
        mnistm_train_x[:split_target].clone(),
        mnistm_train_y[:split_target].clone())
    client2_target = (
        mnistm_train_x[split_target:].clone(),
        mnistm_train_y[split_target:].clone())

    client_cfg = config.clone()
    client1 = _build_client_data(client_cfg.clone(), client1_source,
                                 client1_target)
    client2 = _build_client_data(client_cfg.clone(), client2_source,
                                 client2_target)

    test_dataset = TensorDataset(mnistm_test_x.clone(),
                                 mnistm_test_y.clone())
    server_client = ClientData(client_cfg.clone(),
                               train=None,
                               val=None,
                               test=test_dataset)

    data_dict = {
        0: server_client,
        1: client1,
        2: client2,
    }
    data = StandaloneDataDict(data_dict, config.clone())
    return data, config


register_data('fedmm_mnistm', call_fedmm_mnistm)
