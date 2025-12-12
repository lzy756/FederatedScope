import collections
import copy

import torch
from torch import nn

from federatedscope.core.trainers.torch_trainer import GeneralTorchTrainer
from federatedscope.fedmm.optimizer.pdgd import PDOptimizer
from federatedscope.register import register_trainer


class CyclingBatchSampler:
    def __init__(self, data, labels, batch_size):
        self.data = data
        self.labels = labels
        self.batch_size = batch_size
        self.num_samples = len(data)
        self.order = torch.randperm(self.num_samples) \
            if self.num_samples > 0 else None
        self.offset = 0

    def next(self):
        if self.num_samples == 0 or self.batch_size == 0:
            return None, None
        if self.offset + self.batch_size > self.num_samples:
            self.order = torch.randperm(self.num_samples)
            self.offset = 0
        idx = self.order[self.offset:self.offset + self.batch_size]
        self.offset += self.batch_size
        return self.data[idx], self.labels[idx]


class FedMMTrainer(GeneralTorchTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._param_name_map = {
            id(param): name
            for name, param in self.ctx.model.named_parameters()
        }
        self._dual_state = None
        self._prev_global_state = None
        self._round_idx = 0
        self._domain_criterion = nn.CrossEntropyLoss()

    def register_default_hooks_train(self):
        self.hooks_in_train = collections.defaultdict(list)

    def train(self, target_data_split_name="train", hooks_set=None):
        fedmm_data = getattr(self.ctx.data, 'fedmm_data', None)
        if fedmm_data is None:
            raise ValueError(
                "FedMMTrainer expects `fedmm_data` in client data, but it is "
                "missing.")

        source_x, source_y = fedmm_data['source']
        target_x, target_y = fedmm_data['target']
        total_samples = len(source_x) + len(target_x)
        if total_samples == 0:
            return 0, self.get_model_para(), {}

        device = self.ctx.device
        self.ctx.model.to(device)
        self.ctx.model.train()
        if hasattr(self.ctx.model, 'set_grl_lambda'):
            self.ctx.model.set_grl_lambda(self.cfg.fedmm.domain_loss_coef)

        source_bs = self._calc_source_batch(len(source_x), len(target_x))
        target_bs = self.cfg.fedmm.batch_size - source_bs

        src_iter = CyclingBatchSampler(source_x, source_y,
                                       source_bs) if source_bs > 0 else None
        tgt_iter = CyclingBatchSampler(target_x, target_y,
                                       target_bs) if target_bs > 0 else None

        optimizer = PDOptimizer(self.ctx.model.parameters(),
                                lr=self.cfg.train.optimizer.lr,
                                mu=self.cfg.fedmm.mu)

        if self._prev_global_state is None:
            self._prev_global_state = {
                name: param.detach().clone()
                for name, param in self.ctx.model.named_parameters()
            }

        optimizer.set_reference(self._prev_global_state, self._param_name_map)
        optimizer.set_dual(self._get_dual_state(), self._param_name_map)

        total_loss = 0.
        total_steps = 0
        for _ in range(self.cfg.train.local_update_steps):
            batch = self._prepare_batch(src_iter, tgt_iter, device)
            if batch is None:
                break
            inputs, domain_labels, src_labels, src_count = batch
            logits, domain_logits = self.ctx.model(inputs, return_domain=True)

            cls_loss = torch.tensor(0., device=device)
            if src_labels is not None and src_count > 0:
                cls_loss = self.ctx.criterion(logits[-src_count:], src_labels)
            domain_loss = self._domain_criterion(domain_logits, domain_labels)
            loss = cls_loss + domain_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_steps += 1

        model_state = self.get_model_para()
        shared_state = self._adjust_with_dual(model_state)
        avg_loss = total_loss / total_steps if total_steps > 0 else 0.
        return max(total_samples, 1), shared_state, {'loss': avg_loss}

    def update(self, model_parameters, strict=False):
        super().update(model_parameters, strict)
        self._prev_global_state = {
            name: param.detach().clone()
            for name, param in self.ctx.model.named_parameters()
        }

    def _calc_source_batch(self, source_size, target_size):
        total = source_size + target_size
        if total == 0:
            return self.cfg.fedmm.batch_size
        src_bs = int(
            self.cfg.fedmm.batch_size * source_size / total) if total > 0 else 0
        src_bs = min(self.cfg.fedmm.batch_size, src_bs)
        if source_size > 0 and src_bs == 0:
            src_bs = 1
        if source_size == 0:
            src_bs = 0
        src_bs = min(src_bs, source_size) if source_size > 0 else 0
        src_bs = min(src_bs, self.cfg.fedmm.batch_size)
        return src_bs

    def _prepare_batch(self, src_iter, tgt_iter, device):
        tensors = []
        domain = []
        src_labels = None
        src_count = 0
        if tgt_iter is not None:
            tgt_x, _ = tgt_iter.next()
            if tgt_x is not None:
                tensors.append(tgt_x)
                domain.append(torch.ones(len(tgt_x)))
        if src_iter is not None:
            src_x, src_y = src_iter.next()
            if src_x is not None:
                tensors.append(src_x)
                domain.append(torch.zeros(len(src_x)))
                src_labels = src_y.to(device).long()
                src_count = len(src_x)
        if not tensors:
            return None
        inputs = torch.cat(tensors, dim=0).to(device).float()
        domain_labels = torch.cat(domain).long().to(device)
        return inputs, domain_labels, src_labels, src_count

    def _get_dual_state(self):
        if self._dual_state is None:
            self._dual_state = {
                name: torch.zeros_like(param.detach())
                for name, param in self.ctx.model.named_parameters()
            }
        return self._dual_state

    def _adjust_with_dual(self, model_state):
        if self._prev_global_state is None:
            return model_state
        shared_state = copy.deepcopy(model_state)
        lam = self.cfg.fedmm.lambda1
        decay = self.cfg.fedmm.lambda1_decay ** (
            self._round_idx / max(self.cfg.fedmm.lambda1_decay_step, 1))
        lambdat = lam * decay
        for name, tensor in model_state.items():
            if name not in self._prev_global_state:
                continue
            dual = self._dual_state.get(name)
            if dual is None:
                dual = torch.zeros_like(tensor)
            diff = tensor - self._prev_global_state[name]
            dual = dual + lam * diff
            self._dual_state[name] = dual
            shared_state[name] = tensor + dual / max(lambdat, 1e-12)
        self._round_idx += 1
        return shared_state


def call_fedmm_trainer(trainer_type):
    if trainer_type.lower() == 'fedmm_trainer':
        return FedMMTrainer


register_trainer('fedmm_trainer', call_fedmm_trainer)
