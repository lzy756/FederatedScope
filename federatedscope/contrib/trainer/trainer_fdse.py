from federatedscope.register import register_trainer
from federatedscope.core.trainers.torch_trainer import GeneralTorchTrainer
import os
import logging
import copy
import numpy as np

try:
    import torch
    from torch.utils.data import DataLoader, Dataset
except ImportError:
    torch = None
    DataLoader = None
    Dataset = None

from federatedscope.core.trainers.enums import MODE, LIFECYCLE
from federatedscope.core.trainers.trainer import Trainer
from federatedscope.core.trainers.context import CtxVar
from federatedscope.core.auxiliaries.optimizer_builder import get_optimizer
from federatedscope.core.auxiliaries.scheduler_builder import get_scheduler
from federatedscope.core.data import ClientData
from federatedscope.core.data.wrap_dataset import WrapDataset
from federatedscope.core.auxiliaries.dataloader_builder import get_dataloader
from federatedscope.core.auxiliaries.ReIterator import ReIterator
from federatedscope.core.auxiliaries.utils import param2tensor, merge_param_dict
from federatedscope.core.monitors.monitor import Monitor
from federatedscope.core.trainers.utils import filter_by_specified_keywords

logger = logging.getLogger(__name__)


class FDSETrainer(GeneralTorchTrainer):
    def __init__(self, model, data, device, config, only_for_eval=False, monitor=None):
        super().__init__(model, data, device, config, only_for_eval, monitor)
        self.global_bn_statistics = self.ctx.model.get_bn_dfe_statistics()
        self.custom_bn_statistics = {
            name: {
                "running_mean": torch.zeros_like(
                    stat["running_mean"], device=device
                ).requires_grad_(False),
                "running_var": torch.ones_like(
                    stat["running_var"], device=device
                ).requires_grad_(False),
            }
            for name, stat in self.global_bn_statistics.items()
        }
        # self.global_model = copy.deepcopy(self.ctx.model)

    def _param_filter(self, state_dict, filter_keywords=None):
        """
        model parameter filter when transmit between local and gloabl,
        which is useful in personalization.
        e.g., setting cfg.personalization.local_param= ['bn', 'norms']
        indicates the implementation of
        "FedBN: Federated Learning on Non-IID Features via Local Batch
        Normalization, ICML2021", which can be found in
        https://openreview.net/forum?id=6YEQUn0QICG

        Arguments:
            state_dict (dict): PyTorch Module object's state_dict.
        Returns:
            state_dict (dict): remove the keys that match any of the given
            keywords.
        """
        if self.cfg.federate.method in ["local", "global"]:
            return {}

        if filter_keywords is None:
            filter_keywords = self.cfg.personalization.local_param

        trainable_filter = lambda p: (
            True
            if self.cfg.personalization.share_non_trainable_para
            else p in self.ctx.trainable_para_names
        )
        keyword_filter = filter_by_specified_keywords
        model_para = dict(
            filter(
                lambda elem: trainable_filter(elem[0])
                and keyword_filter(elem[0], filter_keywords),
                state_dict.items(),
            )
        )
        filtered_model_para = {}
        for name, param in model_para.items():
            if "dse.submodules" in name:
                if "running_mean" in name or "running_var" in name:
                    continue

                if "0.weight" in name or "0.bias" in name:
                    continue

            filtered_model_para[name] = param
        return filtered_model_para

    def register_default_hooks_train(self):
        self.register_hook_in_train(self._hook_on_data_parallel_init, "on_fit_start")
        self.register_hook_in_train(self._hook_on_fit_start_init, "on_fit_start")
        self.register_hook_in_train(
            self._hook_on_fit_start_calculate_model_size, "on_fit_start"
        )
        self.register_hook_in_train(self._hook_on_epoch_start, "on_epoch_start")
        self.register_hook_in_train(self._hook_on_batch_start_init, "on_batch_start")
        self.register_hook_in_train(
            self._hook_on_batch_forward_during_train, "on_batch_forward"
        )
        self.register_hook_in_train(
            self._hook_on_batch_forward_regularizer, "on_batch_forward"
        )
        self.register_hook_in_train(
            self._hook_on_batch_forward_flop_count, "on_batch_forward"
        )
        self.register_hook_in_train(self._hook_on_batch_backward, "on_batch_backward")
        self.register_hook_in_train(self._hook_on_batch_end, "on_batch_end")
        self.register_hook_in_train(self._hook_on_fit_end, "on_fit_end")

    def _hook_on_fit_start_init(self, ctx):
        """
        Note:
          The modified attributes and according operations are shown below:
            ==================================  ===========================
            Attribute                           Operation
            ==================================  ===========================
            ``ctx.model``                       Move to ``ctx.device``
            ``ctx.optimizer``                   Initialize by ``ctx.cfg``
            ``ctx.scheduler``                   Initialize by ``ctx.cfg``
            ``ctx.loss_batch_total``            Initialize to 0
            ``ctx.loss_regular_total``          Initialize to 0
            ``ctx.num_samples``                 Initialize to 0
            ``ctx.ys_true``                     Initialize to ``[]``
            ``ctx.ys_prob``                     Initialize to ``[]``
            ==================================  ===========================
        """
        # prepare model and optimizer
        ctx.model.to(ctx.device)

        if ctx.cur_mode in [MODE.TRAIN, MODE.FINETUNE]:
            # Initialize optimizer here to avoid the reuse of optimizers
            # across different routines
            ctx.optimizer = get_optimizer(ctx.model, **ctx.cfg[ctx.cur_mode].optimizer)
            ctx.scheduler = get_scheduler(
                ctx.optimizer, **ctx.cfg[ctx.cur_mode].scheduler
            )
            self.global_bn_statistics = ctx.model.get_bn_dfe_statistics()
            self.global_model = copy.deepcopy(ctx.model)
            self.global_model.eval()

        # TODO: the number of batch and epoch is decided by the current mode
        #  and data split, so the number of batch and epoch should be
        #  initialized at the beginning of the routine

        # prepare statistics
        ctx.loss_batch_total = CtxVar(0.0, LIFECYCLE.ROUTINE)
        ctx.loss_regular_total = CtxVar(0.0, LIFECYCLE.ROUTINE)
        ctx.num_samples = CtxVar(0, LIFECYCLE.ROUTINE)
        ctx.ys_true = CtxVar([], LIFECYCLE.ROUTINE)
        ctx.ys_prob = CtxVar([], LIFECYCLE.ROUTINE)

    def _hook_on_batch_forward_during_train(self, ctx):
        """
        Note:
          The modified attributes and according operations are shown below:
            ==================================  ===========================
            Attribute                           Operation
            ==================================  ===========================
            ``ctx.y_true``                      Move to `ctx.device`
            ``ctx.y_prob``                      Forward propagation get y_prob
            ``ctx.loss_batch``                  Calculate the loss
            ``ctx.batch_size``                  Get the batch_size
            ==================================  ===========================
        """
        x, label = [_.to(ctx.device) for _ in ctx.data_batch]
        pred, des_outputs = ctx.model(x, return_dse_outputs=True)
        global_pred = self.global_model(x)
        if len(label.size()) == 0:
            label = label.unsqueeze(0)

        loss_con_list = []
        layer_names = list(des_outputs.keys())  # 保存层名称以保持顺序
        beta = 0.001  # 控制权重递增速率的超参数，可根据需要调整
        momentum = 0.9
        max_loss_value = 10  # 限制损失值的最大值，防止数值爆炸
        for name in layer_names:
            dse_out = des_outputs[name]
            # 计算批次均值和方差
            batch_mean = dse_out.mean(dim=[0, 2, 3])
            batch_var = dse_out.var(dim=[0, 2, 3], unbiased=False)
            # 使用更平滑的更新规则
            local_running_mean = (
                momentum * self.custom_bn_statistics[name]["running_mean"]
                + (1 - momentum) * batch_mean
            )
            local_running_var = (
                momentum * self.custom_bn_statistics[name]["running_var"]
                + (1 - momentum) * batch_var
            )

            global_running_mean = self.global_bn_statistics[name]["running_mean"]
            global_running_var = self.global_bn_statistics[name]["running_var"]
            # 获取特征维度d（通道数）用于归一化
            d = local_running_mean.shape[0]  # 假设形状为[C]，C是通道数

            # 计算均值的L2损失
            mean_loss = (
                torch.norm(local_running_mean - global_running_mean, p=2) ** 2 / d
            )

            # 计算方差的L1损失
            # local_var_l1 = torch.norm(local_running_var, p=1)
            # global_var_l1 = torch.norm(global_running_var, p=1)
            # var_loss = ((local_var_l1 - global_var_l1) / d) ** 2
            # 计算方差的L2损失
            var_loss = torch.norm(local_running_var - global_running_var, p=2) ** 2 / d
            # 计算当前层的一致性损失
            layer_loss_con = mean_loss + var_loss

            # 限制损失值防止数值爆炸
            layer_loss_con = torch.clamp(layer_loss_con, max=max_loss_value)

            # 检查是否为NaN或无穷大
            if torch.isnan(layer_loss_con) or torch.isinf(layer_loss_con):
                logger.warning(f"Invalid loss detected for layer {name}, setting to 0")
                layer_loss_con = torch.tensor(
                    0.0, device=ctx.device, requires_grad=True
                )

            loss_con_list.append(layer_loss_con)

            self.custom_bn_statistics[name]["running_mean"] = (
                local_running_mean.detach().clone().requires_grad_(False)
            )
            self.custom_bn_statistics[name]["running_var"] = (
                local_running_var.detach().clone().requires_grad_(False)
            )

        # 计算总一致性损失（使用指数权重的多层聚合）
        if loss_con_list:
            L = len(loss_con_list)
            layer_indices = torch.tensor(
                [i + 1 for i in range(L)], device=ctx.device, dtype=torch.float32
            )

            exp_weights = torch.exp(beta * layer_indices)
            exp_weights = exp_weights / exp_weights.sum()
            loss_con_values = torch.stack(loss_con_list).to(exp_weights.dtype)
            loss_con = torch.sum(loss_con_values * exp_weights)
        else:
            loss_con = torch.tensor(0.0, device=ctx.device)

        logger.info(
            f"Consistency loss: {loss_con.item()}, list: {[l.item() for l in loss_con_list]}"
        )

        # 计算KL散度损失
        kl_loss_fn = torch.nn.KLDivLoss(reduction="batchmean")
        kl_loss = kl_loss_fn(
            torch.log_softmax(pred, dim=1), torch.softmax(global_pred, dim=1)
        )
        logger.info(f"KL divergence loss: {kl_loss.item()}")
        ctx.y_true = CtxVar(label, LIFECYCLE.BATCH)
        ctx.y_prob = CtxVar(pred, LIFECYCLE.BATCH)
        ctx.loss_batch = CtxVar(
            ctx.criterion(pred, label) + loss_con + 0.01 * kl_loss, LIFECYCLE.BATCH
        )
        ctx.batch_size = CtxVar(len(label), LIFECYCLE.BATCH)


def call_fdse_trainer(trainer_type):
    if trainer_type == "fdsetrainer":
        trainer_builder = FDSETrainer
        return trainer_builder


register_trainer("fdsetrainer", call_fdse_trainer)
