import math
import os
import pickle

import torch
from torch.utils.data import TensorDataset
from torchvision.datasets import MNIST

from federatedscope.core.splitters.utils import \
    dirichlet_distribution_noniid_slice
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


def _normalize_domain(value):
    if value is None:
        raise ValueError("Domain entry requires `domain` to be specified.")
    domain = str(value).strip().lower()
    if domain in ['source', 'src', 's']:
        return 'source'
    if domain in ['target', 'tgt', 't']:
        return 'target'
    raise ValueError(f"Unrecognized domain spec `{value}`.")


def _normalize_domain_groups(config):
    groups = getattr(config.fedmm, 'domain_groups', None)
    if not groups:
        return None
    normalized = []
    total_clients = 0
    seen_domains = set()
    for raw in groups:
        domain = _normalize_domain(_maybe_get(raw, 'domain'))
        if domain in seen_domains:
            raise ValueError(
                "Only one group per domain is supported in "
                "`fedmm.domain_groups`. Found duplicate for "
                f"`{domain}`.")
        client_num = int(_maybe_get(raw, 'client_num', 0))
        if client_num <= 0:
            continue
        splitter = _maybe_get(raw, 'splitter', 'uniform')
        if splitter is None:
            splitter = 'uniform'
        splitter = str(splitter).strip().lower()
        splitter_args = _maybe_get(raw, 'splitter_args', {}) or {}
        alpha = _maybe_get(raw, 'alpha')
        if alpha is not None:
            splitter_args = dict(splitter_args)
            splitter_args.setdefault('alpha', alpha)
        normalized.append({
            'domain': domain,
            'client_num': client_num,
            'splitter': splitter,
            'splitter_args': splitter_args,
        })
        total_clients += client_num
        seen_domains.add(domain)
    if not normalized:
        return None
    if total_clients != config.federate.client_num:
        raise ValueError(
            "Sum of `client_num` in `fedmm.domain_groups` must equal "
            "`federate.client_num`.")
    return normalized


def _empty_like_data(data_x, data_y):
    shape = (0, ) + tuple(data_x.shape[1:])
    empty_x = data_x.new_zeros(shape)
    empty_y = data_y.new_zeros((0, ), dtype=data_y.dtype)
    return empty_x, empty_y


def _slice_tensor_pair(data_x, data_y, indices):
    if len(indices) == 0:
        return _empty_like_data(data_x, data_y)
    idx_tensor = torch.tensor(indices, dtype=torch.long)
    return data_x[idx_tensor].clone(), data_y[idx_tensor].clone()


def _split_with_lda(data_x, data_y, client_num, alpha):
    labels = data_y.detach().cpu().numpy()
    idx_slice = dirichlet_distribution_noniid_slice(labels, client_num, alpha)
    return [_slice_tensor_pair(data_x, data_y, idxs) for idxs in idx_slice]


def _split_uniform(data_x, data_y, client_num, domain_name):
    if data_x.size(0) == 0:
        return [_empty_like_data(data_x, data_y) for _ in range(client_num)]
    perm = torch.randperm(data_x.size(0))
    shuffled_x = data_x[perm]
    shuffled_y = data_y[perm]
    weights = [1.0] * client_num
    return _split_tensor_pair(shuffled_x, shuffled_y, weights, domain_name)


def _split_domain_samples(data_x, data_y, group, domain_name):
    splitter = group.get('splitter', 'uniform')
    splitter_args = group.get('splitter_args', {}) or {}
    client_num = group['client_num']
    if splitter == 'lda':
        alpha = float(splitter_args.get('alpha', 0.5))
        return _split_with_lda(data_x, data_y, client_num, alpha)
    else:
        weights = splitter_args.get('weights')
        if weights is not None:
            if len(weights) != client_num:
                raise ValueError(
                    f"`weights` length must match client_num for {domain_name}.")
            perm = torch.randperm(data_x.size(0)) if data_x.size(0) > 0 \
                else None
            shuffled_x = data_x[perm] if perm is not None else data_x
            shuffled_y = data_y[perm] if perm is not None else data_y
            return _split_tensor_pair(shuffled_x, shuffled_y, weights,
                                      domain_name)
        return _split_uniform(data_x, data_y, client_num, domain_name)


def _build_domain_group_clients(config, groups, mnist_source, mnist_source_y,
                                mnistm_target, mnistm_target_y):
    clients = {}
    next_client_id = 1
    empty_source = _empty_like_data(mnist_source, mnist_source_y)
    empty_target = _empty_like_data(mnistm_target, mnistm_target_y)
    for group in groups:
        domain = group['domain']
        if domain == 'source':
            splits = _split_domain_samples(mnist_source, mnist_source_y, group,
                                           'source')
            for split in splits:
                client_cfg = config.clone()
                clients[next_client_id] = _build_client_data(
                    client_cfg, split, empty_target)
                next_client_id += 1
        else:
            splits = _split_domain_samples(mnistm_target, mnistm_target_y,
                                           group, 'target')
            for split in splits:
                client_cfg = config.clone()
                clients[next_client_id] = _build_client_data(
                    client_cfg, empty_source, split)
                next_client_id += 1
    return clients


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

    domain_groups = _normalize_domain_groups(config)
    specs = None
    if domain_groups is None:
        specs = _normalize_specs(config)
    if domain_groups is None and specs is None \
            and config.federate.client_num != 2:
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
    if domain_groups is not None:
        client_data = _build_domain_group_clients(config, domain_groups,
                                                  mnist_train_x, mnist_train_y,
                                                  mnistm_train_x,
                                                  mnistm_train_y)
    elif specs is not None:
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
