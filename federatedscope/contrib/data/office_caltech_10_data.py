import os
import json
import random
import logging

import numpy as np
import torch
from torch.utils.data import Subset, TensorDataset
from torchvision import datasets, transforms
from scipy.stats import entropy
from sklearn.cluster import DBSCAN

from federatedscope.core.data.base_data import StandaloneDataDict, ClientData
from federatedscope.register import register_data
from federatedscope.contrib.data.utils import (
    dirichlet_split,
    get_label_distribution,
    adaptive_cluster_clients,
    save_cluster_results,
)

logger = logging.getLogger(__name__)


def load_office_caltech10(config, client_cfgs=None):
    data_root = config.data.root
    alpha = config.data.splitter_args[0].get("alpha", 0.5)
    total_clients = config.federate.get("client_num", 100)

    # train-transform
    office_transforms = transforms.Compose(
        [
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
        ]
    )

    # 四个域
    domains = ["amazon", "caltech", "dslr", "webcam"]
    train_ds = []
    test_ds = []

    for dm in domains:
        root = os.path.join(data_root, dm)
        # train dataset
        total_sample = datasets.ImageFolder(root, transform=office_transforms)
        train_size = int(0.8 * len(total_sample))
        test_size = len(total_sample) - train_size
        train_dataset, test_dataset = torch.utils.data.random_split(
            total_sample, [train_size, test_size]
        )
        train_ds.append(train_dataset)
        test_ds.append(test_dataset)
        logger.info(
            f"Domain {dm}: train size = {len(train_dataset)}, test size = {len(test_dataset)}"
        )

    # flatten train splits
    train_splits = []
    client_for_each_domain = [4, 4, 1, 1]  # 每个域的客户端数量
    for ds, num in zip(train_ds, client_for_each_domain):
        parts = dirichlet_split(ds, num, alpha)
        train_splits += [Subset(ds, p) for p in parts]

    test_splits = []
    for ds, num in zip(test_ds, client_for_each_domain):
        parts = dirichlet_split(ds, num, alpha)
        test_splits += [Subset(ds, p) for p in parts]

    # distributions
    train_dists = get_label_distribution(train_splits, num_classes=10)
    test_dists = get_label_distribution(test_splits, num_classes=10)

    # clustering
    communities = adaptive_cluster_clients(train_dists)
    logger.info(f"Peer communities: {communities}")
    save_cluster_results(
        os.path.join(data_root, "peer_communities.json"),
        train_dists,
        test_dists,
        communities,
    )

    # pack into FederatedScope
    cd = {}
    for cid, (tr, te) in enumerate(zip(train_splits, test_splits), start=1):
        cd[cid] = ClientData(client_cfg=config, train=tr, val=None, test=te)
    fsdata = StandaloneDataDict(cd, config)
    return fsdata, config


def call_office_caltech10_data(config, client_cfgs=None):
    if config.data.type == "office_caltech10":
        return load_office_caltech10(config, client_cfgs)


if __name__ == "__main__":
    from federatedscope.core.configs.config import global_cfg

    global_cfg.data.root = os.path.expanduser(
        "/root/project/FederatedScope/data/office_caltech_10"
    )
    global_cfg.data.type = "office_caltech10"
    global_cfg.data.splitter_args = [{"alpha": 0.5}]
    global_cfg.federate.client_num = 100
    data, cfg = load_office_caltech10(global_cfg)
else:
    register_data("office_caltech10", call_office_caltech10_data)
