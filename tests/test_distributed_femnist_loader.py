import json
import os
import tempfile
import unittest

import torch

from federatedscope.core.auxiliaries.data_builder import get_data
from federatedscope.core.configs.config import global_cfg


class DistributedFemnistLoaderTest(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.root = self.tmpdir.name
        self.shard_dir = os.path.join(self.root, 'client_1')
        os.makedirs(self.shard_dir, exist_ok=True)
        self._prepare_split('train', num_samples=8)
        self._prepare_split('val', num_samples=2)
        self._prepare_split('test', num_samples=3)
        meta = {
            'client_id': 1,
            'tasks': [0, 1],
            'num_samples': {
                'train': 8,
                'val': 2,
                'test': 3
            },
            'label_hist': {
                '0': 4,
                '1': 4
            }
        }
        with open(os.path.join(self.shard_dir, 'meta.json'), 'w') as f:
            json.dump(meta, f)
        manifest = {
            'dataset': 'femnist',
            'clients': [{
                'client_id': 1,
                'shard_path': self.shard_dir
            }]
        }
        self.manifest_path = os.path.join(self.root, 'manifest.json')
        with open(self.manifest_path, 'w') as f:
            json.dump(manifest, f)

    def tearDown(self):
        self.tmpdir.cleanup()

    def _prepare_split(self, name, num_samples):
        data = torch.randint(0, 255, (num_samples, 28 * 28), dtype=torch.uint8)
        targets = torch.randint(0, 62, (num_samples, ), dtype=torch.long)
        torch.save((data, targets), os.path.join(self.shard_dir, f'{name}.pt'))

    def test_loader_builds_client_data(self):
        cfg = global_cfg.clone()
        cfg.federate.mode = 'distributed'
        cfg.distribute.use = True
        cfg.distribute.role = 'client'
        cfg.distribute.server_host = '127.0.0.1'
        cfg.distribute.server_port = 50051
        cfg.distribute.client_host = '127.0.0.1'
        cfg.distribute.client_port = 50052
        cfg.distribute.client_id = 1
        cfg.distributed_data.enabled = True
        cfg.distributed_data.dataset = 'femnist'
        cfg.distributed_data.shard_path = self.shard_dir
        cfg.distributed_data.manifest = self.manifest_path
        cfg.distributed_data.strict_check = True
        cfg.distributed_data.meta_file = 'meta.json'
        cfg.data.type = 'femnist'
        cfg.data.root = self.root
        cfg.data.transform = [['ToTensor']]
        cfg.dataloader.batch_size = 4
        cfg.federate.client_num = 3

        data, _ = get_data(cfg.clone())
        self.assertIsNotNone(data)
        self.assertTrue(hasattr(data, 'metadata'))
        self.assertEqual(data.metadata['num_samples']['train'], 8)
        self.assertEqual(len(data.train_data), 8)
        train_loader = data['train']
        batch = next(iter(train_loader))
        self.assertEqual(batch[0].shape[0], cfg.dataloader.batch_size)


if __name__ == '__main__':
    unittest.main()
