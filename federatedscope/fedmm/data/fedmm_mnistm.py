import math
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


def _maybe_get(obj, key, default=None):
    if isinstance(obj, dict):
        return obj.get(key, default)
    if hasattr(obj, key):
        return getattr(obj, key)
    getter = getattr(obj, 'get', None)
    if callable(getter):
        try:
            return getter(key, default)
        except TypeError:
            return default
    return default


def _parse_domain_spec(entry, default_domain=None):
    if entry is None:
        return {'source': 0.0, 'target': 0.0}
    if isinstance(entry, str):
        default_domain = entry
        entry = {}
    if default_domain is None:
        default_domain = _maybe_get(entry, 'domain')
    source_weight = _maybe_get(entry, 'source')
    if source_weight is None:
        source_weight = _maybe_get(entry, 'source_weight')
    if source_weight is None:
        source_weight = _maybe_get(entry, 'source_ratio')
    target_weight = _maybe_get(entry, 'target')
    if target_weight is None:
        target_weight = _maybe_get(entry, 'target_weight')
    if target_weight is None:
        target_weight = _maybe_get(entry, 'target_ratio')
    if source_weight is None and default_domain:
        domain = str(default_domain).strip().lower()
        if domain in ['source', 'src']:
            source_weight = 1.0
            target_weight = 0.0 if target_weight is None else target_weight
        elif domain in ['target', 'tgt']:
            target_weight = 1.0
            source_weight = 0.0 if source_weight is None else source_weight
        elif domain in ['both', 'all', 'mixed']:
            source_weight = 1.0
            target_weight = 1.0
    source_weight = 0.0 if source_weight is None else float(source_weight)
    target_weight = 0.0 if target_weight is None else float(target_weight)
    source_weight = max(0.0, source_weight)
    target_weight = max(0.0, target_weight)
    return {'source': source_weight, 'target': target_weight}


def _normalize_specs(config):
    specs = getattr(config.fedmm, 'client_domain_specs', None)
    if not specs:
        return None
    if len(specs) != config.federate.client_num:
        raise ValueError(
            "Length of `fedmm.client_domain_specs` must equal "
            "`federate.client_num` when provided.")
    normalized = []
    for entry in specs:
        normalized.append(_parse_domain_spec(entry))
    return normalized


def _calc_counts(total_size, weights, domain_name):
    if total_size == 0:
        return [0] * len(weights)
    total_weight = sum(max(0.0, float(w)) for w in weights)
    if total_weight <= 0.0:
        raise ValueError(
            f"No positive allocation weight provided for {domain_name} data "
            f"while the dataset contains samples.")
    normalized = [max(0.0, float(w)) / total_weight for w in weights]
    desired = [w * total_size for w in normalized]
    counts = [int(math.floor(val)) for val in desired]
    remainder = total_size - sum(counts)
    if remainder > 0:
        fracs = [(desired[i] - counts[i], i) for i in range(len(counts))]
        fracs.sort(key=lambda item: item[0], reverse=True)
        for _, idx in fracs[:remainder]:
            counts[idx] += 1
    return counts


def _split_tensor_pair(data_x, data_y, weights, domain_name):
    splits = []
    total = data_x.size(0)
    counts = _calc_counts(total, weights, domain_name)
    start = 0
    for count in counts:
        end = start + count
        splits.append((data_x[start:end].clone(), data_y[start:end].clone()))
        start = end
    return splits


def _build_multi_client_data(config, specs, mnist_source, mnist_source_y,
                             mnistm_target, mnistm_target_y):
    source_weights = [spec['source'] for spec in specs]
    target_weights = [spec['target'] for spec in specs]
    source_splits = _split_tensor_pair(mnist_source, mnist_source_y,
                                       source_weights, 'source')
    target_splits = _split_tensor_pair(mnistm_target, mnistm_target_y,
                                       target_weights, 'target')
    clients = {}
    for idx, _ in enumerate(specs, start=1):
        client_cfg = config.clone()
        client = _build_client_data(client_cfg, source_splits[idx - 1],
                                    target_splits[idx - 1])
        clients[idx] = client
    return clients


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

    specs = _normalize_specs(config)
    if specs is None and config.federate.client_num != 2:
        raise ValueError(
            "FedMM requires `fedmm.client_domain_specs` to build data for "
            "more than 2 clients.")

    root = config.data.root
    mnist_train_x, mnist_train_y, mnist_test_x, mnist_test_y = _load_mnist(
        root)
    mnistm_train_x, mnistm_test_x = _load_mnistm(root)

    # MNIST-M labels follow the original MNIST labels
    mnistm_train_y = mnist_train_y.clone()
    mnistm_test_y = mnist_test_y.clone()

    num_source = mnist_train_x.size(0)
    num_target = mnistm_train_x.size(0)

    client_cfg = config.clone()
    if specs is not None:
        client_data = _build_multi_client_data(config, specs, mnist_train_x,
                                               mnist_train_y, mnistm_train_x,
                                               mnistm_train_y)
    else:
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

        client_data = {
            1:
            _build_client_data(client_cfg.clone(), client1_source,
                               client1_target),
            2:
            _build_client_data(client_cfg.clone(), client2_source,
                               client2_target),
        }

    test_dataset = TensorDataset(mnistm_test_x.clone(),
                                 mnistm_test_y.clone())
    server_client = ClientData(client_cfg.clone(),
                               train=None,
                               val=None,
                               test=test_dataset)

    data_dict = {0: server_client}
    data_dict.update(client_data)
    data = StandaloneDataDict(data_dict, config.clone())
    return data, config


register_data('fedmm_mnistm', call_fedmm_mnistm)
