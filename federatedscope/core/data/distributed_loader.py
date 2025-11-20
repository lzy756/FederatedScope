import json
import logging
import os.path as osp
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from federatedscope.core.auxiliaries.transform_builder import get_transform
from federatedscope.core.auxiliaries.manifest_utils import (
    find_client_in_manifest, load_manifest, summarize_manifest)
from federatedscope.core.data.base_data import ClientData

logger = logging.getLogger(__name__)

_SUPPORTED_DATASETS = {'femnist'}
_IMAGE_SIZE = {'femnist': (28, 28)}
_MODE = {'femnist': 'L'}


class FemnistShardDataset(Dataset):
    """Lightweight dataset that mirrors LEAF_CV behavior for FEMNIST."""
    def __init__(self,
                 data_tensor: torch.Tensor,
                 target_tensor: torch.Tensor,
                 transform=None,
                 target_transform=None):
        self.data = data_tensor
        self.targets = target_tensor
        self.transform = transform
        self.target_transform = target_transform
        self.image_size = _IMAGE_SIZE['femnist']
        self.mode = _MODE['femnist']

    def __len__(self):
        return 0 if self.targets is None else self.targets.shape[0]

    def __getitem__(self, index):
        img = self.data[index].clone().cpu().numpy()
        if img.ndim == 1:
            img = img.reshape(self.image_size)
        img = Image.fromarray(img.astype(np.uint8), mode=self.mode)
        if self.transform:
            img = self.transform(img)
        else:
            tensor = torch.from_numpy(np.array(img, copy=True)).float()
            img = tensor.unsqueeze(0) / 255.0
        target = self.targets[index]
        if isinstance(target, torch.Tensor):
            target = target.clone()
        if self.target_transform:
            target = self.target_transform(target)
        elif isinstance(target, torch.Tensor):
            target = int(target.item())
        return img, target


class DistributedShardLoader:
    """Loader that reads per-client FEMNIST shards from disk."""
    def __init__(self, config):
        self.config = config
        self.client_id = config.distribute.client_id
        self.shard_path = osp.abspath(config.distributed_data.shard_path)
        self.strict = config.distributed_data.strict_check
        self.meta_file = config.distributed_data.meta_file or 'meta.json'
        self.dataset_name = (config.distributed_data.dataset
                             or config.data.type).lower()
        if self.dataset_name not in _SUPPORTED_DATASETS:
            raise NotImplementedError(
                f"Distributed shard loader currently supports "
                f"{_SUPPORTED_DATASETS}, but got dataset type "
                f"'{self.dataset_name}'.")
        self.manifest = load_manifest(
            config.distributed_data.manifest,
            strict=config.distributed_data.strict_check)

    def build_client_data(self) -> ClientData:
        self._ensure_exists(self.shard_path)
        transforms, val_transforms, test_transforms = \
            get_transform(self.config, 'torchvision')
        train_pair = self._load_split('train')
        val_pair = self._load_split('val', optional=True)
        test_pair = self._load_split('test', optional=True)
        train_ds = self._wrap_pair(train_pair, transforms)
        val_ds = self._wrap_pair(val_pair, val_transforms)
        test_ds = self._wrap_pair(test_pair, test_transforms)
        if train_ds is None:
            raise ValueError(
                f"No train split found under {self.shard_path}. Each shard "
                "must provide at least train.pt")
        client_data = ClientData(self.config,
                                 train=train_ds,
                                 val=val_ds,
                                 test=test_ds)
        metadata = self._build_metadata(train_ds, val_ds, test_ds)
        setattr(client_data, 'metadata', metadata)
        logger.info(
            'Client #%s loaded FEMNIST shard from %s (train=%s, val=%s, '
            'test=%s)', self.client_id, self.shard_path,
            len(train_ds) if train_ds else 0,
            len(val_ds) if val_ds else 0,
            len(test_ds) if test_ds else 0)
        return client_data

    def _load_split(
            self,
            split: str,
            optional: bool = False
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        file_path = osp.join(self.shard_path, f'{split}.pt')
        if not osp.exists(file_path):
            if optional or not self.strict:
                if not optional:
                    logger.warning('%s missing %s split at %s',
                                   self.shard_path, split, file_path)
                return None
            raise FileNotFoundError(
                f"Expected {split}.pt under {self.shard_path} but not found")
        data, targets = torch.load(file_path)
        return data, targets

    def _wrap_pair(self, pair, transform_cfg: Optional[Dict[str, Any]]):
        if pair is None:
            return None
        transform = transform_cfg.get('transform') if transform_cfg else None
        target_transform = None
        if transform_cfg:
            target_transform = transform_cfg.get('target_transform')
        data_tensor, target_tensor = pair
        return FemnistShardDataset(data_tensor, target_tensor, transform,
                                   target_transform)

    def _build_metadata(self, train_ds, val_ds, test_ds):
        manifest_entry = find_client_in_manifest(self.manifest, self.client_id,
                                                 self.shard_path)
        meta = self._read_meta_file()
        if manifest_entry:
            meta.setdefault('manifest', manifest_entry)
        meta['client_id'] = self.client_id
        meta['shard_path'] = self.shard_path
        meta['num_samples'] = {
            'train': len(train_ds) if train_ds else 0,
            'val': len(val_ds) if val_ds else 0,
            'test': len(test_ds) if test_ds else 0
        }
        return meta

    def _read_meta_file(self):
        meta_path = osp.join(self.shard_path, self.meta_file)
        if not osp.exists(meta_path):
            return {}
        try:
            with open(meta_path, 'r') as f:
                return json.load(f)
        except json.JSONDecodeError as err:
            logger.warning('Failed to parse %s: %s', meta_path, err)
            return {}

    def _ensure_exists(self, path):
        if not osp.isdir(path):
            raise FileNotFoundError(
                f"Shard directory {path} not found. Please run the "
                f"scripts/tools/prepare_femnist_shards.py utility first.")


def load_distributed_data(config):
    if config.distribute.role == 'server':
        manifest = load_manifest(config.distributed_data.manifest,
                                 strict=config.distributed_data.strict_check)
        if manifest:
            logger.info('Loaded manifest summary: %s',
                        summarize_manifest(manifest))
        return None, config
    loader = DistributedShardLoader(config)
    data = loader.build_client_data()
    return data, config
