# Copyright (c) Alibaba, Inc. and its affiliates.
import unittest

from federatedscope.core.auxiliaries.data_builder import get_data
from federatedscope.core.auxiliaries.utils import setup_seed
from federatedscope.core.auxiliaries.logging import update_logger
from federatedscope.core.configs.config import global_cfg
from federatedscope.core.auxiliaries.runner_builder import get_runner
from federatedscope.core.auxiliaries.worker_builder import (
    get_server_cls,
    get_client_cls,
)


class ResnetCifar100Test(unittest.TestCase):
    def setUp(self):
        print(("Testing %s.%s" % (type(self).__name__, self._testMethodName)))

    def set_config_resnet_cifar100(self, cfg):
        backup_cfg = cfg.clone()

        import torch

        cfg.use_gpu = torch.cuda.is_available()
        cfg.eval.freq = 10
        cfg.eval.split = ["test"]

        cfg.eval.metrics = [
            "loss",
            "acc",
        ]
        cfg.eval.best_res_update_round_wise_key = "test_acc"
        cfg.eval.count_flops = False

        cfg.outdir = "./log"
        cfg.federate.mode = "standalone"
        cfg.train.local_update_steps = 10
        cfg.federate.total_round_num = 400
        cfg.federate.sample_client_num = 20
        cfg.federate.client_num = 20

        # cfg.data.root = "test_data/"
        cfg.data.type = "cifar100"
        cfg.data.root = "./cifar100/data"
        cfg.data.splitter = "iid"  # iid
        cfg.dataloader.batch_size = 512

        cfg.wandb.use = True
        cfg.wandb.name_user = "bo-wang-bupt"
        cfg.wandb.name_project = "resnet50_cifar100"

        # cfg.data.args = [{}]
        # cfg.data.splits = [0.6, 0.2, 0.2]
        # cfg.data.subsample = 0.01
        # cfg.data.transform = [
        #     ["ToTensor"],
        #     ["Normalize", {"mean": [0.9637], "std": [0.1592]}],
        # ]
        # cfg.model = "resnet"
        cfg.model.type = "resnet_50"
        cfg.model.num_classes = 100
        # cfg.model.hidden = 2048
        # cfg.model.out_channels = 62
        cfg.train.optimizer.type = "SGD"
        cfg.train.optimizer.momentum = 0.9
        cfg.train.optimizer.lr = 0.01

        # cfg.train.
        # cfg.train.optimizer.weight_decay = 0.0
        cfg.train.scheduler.type = "CosineAnnealingLR"
        cfg.train.scheduler.T_max = 20

        # cfg.train.scheduler.T_max =
        cfg.criterion.type = "CrossEntropyLoss"
        cfg.trainer.type = "cvtrainer"
        cfg.seed = 123

        # cfg.fedprox.use = True
        # cfg.fedprox.mu = 0.1

        return backup_cfg

    def test_resnet_cifar100_standalone(self):
        init_cfg = global_cfg.clone()
        backup_cfg = self.set_config_resnet_cifar100(init_cfg)
        setup_seed(init_cfg.seed)
        update_logger(init_cfg, True)

        data, modified_cfg = get_data(init_cfg.clone())
        init_cfg.merge_from_other_cfg(modified_cfg)
        self.assertIsNotNone(data)

        Fed_runner = get_runner(
            data=data,
            server_class=get_server_cls(init_cfg),
            client_class=get_client_cls(init_cfg),
            config=init_cfg.clone(),
        )
        self.assertIsNotNone(Fed_runner)
        test_results = Fed_runner.run()
        init_cfg.merge_from_other_cfg(backup_cfg)

        # self.assertLess(
        #     test_results["client_summarized_weighted_avg"]["test_loss"], 600
        # )


if __name__ == "__main__":
    unittest.main()
