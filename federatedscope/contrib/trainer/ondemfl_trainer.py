from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    import torch
    import torch.nn as nn
else:
    try:
        import torch
        import torch.nn as nn
    except ImportError:  # pragma: no cover
        torch = None
        nn = None

import logging

from yacs.config import CfgNode

from federatedscope.core.auxiliaries.optimizer_builder import get_optimizer
from federatedscope.core.trainers.context import CtxVar
from federatedscope.core.trainers.enums import LIFECYCLE
from federatedscope.core.trainers.torch_trainer import GeneralTorchTrainer
from federatedscope.register import register_trainer


def _require_torch():
    if torch is None or nn is None:
        raise ImportError(
            "PyTorch is required to use the OnDemFL trainer. Please install "
            "torch/torchvision in the current environment.")


def _find_last_linear(module: nn.Module) -> Optional[nn.Linear]:
    last_linear = None
    for sub_module in module.modules():
        if isinstance(sub_module, nn.Linear):
            last_linear = sub_module
    return last_linear


class OnDemFLTrainer(GeneralTorchTrainer):
    """
    Customized trainer that jointly optimizes a backbone model and the class
    distribution predictor required by On-Demand Federated Learning.
    """
    def __init__(self,
                 model=None,
                 data=None,
                 device=None,
                 config=None,
                 only_for_eval=False,
                 monitor=None):
        _require_torch()
        super().__init__(model=model,
                         data=data,
                         device=device,
                         config=config,
                         only_for_eval=only_for_eval,
                         monitor=monitor)
        self._phase = 'pretrain'
        self._cached_linear: Optional[nn.Linear] = None

    def set_phase(self, phase: str):
        assert phase in ['pretrain', 'ondemand'], \
            f'Invalid phase {phase} for OnDemFLTrainer'
        self._phase = phase

    def register_default_hooks_train(self):
        super().register_default_hooks_train()
        self.replace_hook_in_train(self._hook_on_fit_start_init_ondem,
                                   target_trigger='on_fit_start',
                                   target_hook_name='_hook_on_fit_start_init')
        self.replace_hook_in_train(self._hook_on_batch_backward_ondem,
                                   target_trigger='on_batch_backward',
                                   target_hook_name='_hook_on_batch_backward')
        self.register_hook_in_train(self._hook_on_fit_end_ondem,
                                    trigger='on_fit_end',
                                    insert_mode='before')
        self.register_hook_in_train(self._hook_on_fit_end_cleanup,
                                    trigger='on_fit_end',
                                    insert_mode='after')

    def _hook_on_fit_start_init_ondem(self, ctx):
        _require_torch()
        super()._hook_on_fit_start_init(ctx)

        predictor: nn.Module = ctx.model.distribution_predictor
        dp_cfg: CfgNode = self.cfg.ondemfl.dp_optimizer.clone()
        dp_kwargs = {key: dp_cfg[key] for key in dp_cfg}
        dp_type = dp_kwargs.pop('type')
        dp_lr = dp_kwargs.pop('lr')
        if self._phase == 'pretrain' or not \
                self.cfg.ondemfl.freeze_predictor_after_stage1:
            ctx.dp_optimizer = get_optimizer(predictor, dp_type, dp_lr,
                                             **dp_kwargs)
        else:
            ctx.dp_optimizer = None

        loss_name = self.cfg.ondemfl.dp_loss.lower()
        if loss_name == 'mse':
            ctx.dp_loss_fn = nn.MSELoss()
        elif loss_name == 'mae':
            ctx.dp_loss_fn = nn.L1Loss()
        elif loss_name in ['kldiv', 'kl']:
            ctx.dp_loss_fn = nn.KLDivLoss(reduction='batchmean')
        else:
            raise ValueError(f'Unsupported dp_loss `{loss_name}`')

        ctx.dist_pred_sum = CtxVar(
            torch.zeros(predictor.net[-1].out_features, device=ctx.device),
            LIFECYCLE.ROUTINE)
        ctx.dist_pred_count = CtxVar(0, LIFECYCLE.ROUTINE)
        ctx.last_dp_loss = None

    def _hook_on_batch_backward_ondem(self, ctx):
        _require_torch()
        ctx.optimizer.zero_grad()
        ctx.loss_task.backward()

        grad_vec = self._extract_gradient_vector(ctx)
        num_classes = ctx.model.distribution_predictor.net[-1].out_features
        target_dist = self._batch_distribution(ctx.y_true, num_classes,
                                               ctx.device)

        predictor = ctx.model.distribution_predictor

        if ctx.dp_optimizer is not None:
            ctx.dp_optimizer.zero_grad()
            pred_dist = predictor(grad_vec.unsqueeze(0))
            if isinstance(ctx.dp_loss_fn, nn.KLDivLoss):
                dp_loss = ctx.dp_loss_fn(pred_dist.log(), target_dist)
            else:
                dp_loss = ctx.dp_loss_fn(pred_dist, target_dist)
            dp_loss.backward()
            ctx.dp_optimizer.step()
            ctx.last_dp_loss = dp_loss.detach().item()
            pred_detach = pred_dist.detach()
        else:
            with torch.no_grad():
                pred_detach = predictor(grad_vec.unsqueeze(0))
            ctx.last_dp_loss = None

        ctx.dist_pred_sum += pred_detach.squeeze(0)
        ctx.dist_pred_count += 1

        if ctx.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(ctx.model.task_model.parameters(),
                                           ctx.grad_clip)
        ctx.optimizer.step()
        if ctx.scheduler is not None:
            ctx.scheduler.step()

    def _hook_on_fit_end_ondem(self, ctx):
        if ctx.dist_pred_count > 0:
            avg_pred = ctx.dist_pred_sum / ctx.dist_pred_count
            ctx.ondem_pred_distribution = avg_pred.detach().cpu()
        else:
            ctx.ondem_pred_distribution = None

        ctx.ondem_true_distribution = self._compute_true_distribution(ctx)

    @staticmethod
    def _release_optimizer(optimizer) -> bool:
        if optimizer is None:
            return False
        if hasattr(optimizer, 'state'):
            optimizer.state.clear()
        return True

    def _hook_on_fit_end_cleanup(self, ctx):
        released = False

        dp_opt = getattr(ctx, 'dp_optimizer', None)
        if dp_opt is not None:
            released |= self._release_optimizer(dp_opt)
        if 'dp_optimizer' in ctx:
            del ctx['dp_optimizer']

        main_opt = getattr(ctx, 'optimizer', None)
        if main_opt is not None:
            released |= self._release_optimizer(main_opt)
        if 'optimizer' in ctx:
            del ctx['optimizer']

        scheduler = getattr(ctx, 'scheduler', None)
        if scheduler is not None:
            released = True
        if 'scheduler' in ctx:
            del ctx['scheduler']

        model = getattr(ctx, 'model', None)
        if model is not None:
            try:
                model.zero_grad(set_to_none=True)
            except TypeError:
                model.zero_grad()
                for param in model.parameters():
                    if hasattr(param, 'grad') and param.grad is not None:
                        param.grad = None
            released = True

        if released and torch is not None and torch.cuda.is_available():
            device = getattr(ctx, 'device', None)
            device_type = getattr(device, 'type', None)
            if device_type is None and device is not None:
                device_type = str(device)
            if isinstance(device_type, str) and 'cuda' in device_type:
                torch.cuda.empty_cache()

    def _compute_true_distribution(self, ctx) -> Optional[torch.Tensor]:
        dataset = ctx.train_loader if ctx.get('train_loader') is not None \
            else ctx.train_data
        if dataset is None:
            return None

        if hasattr(ctx, 'ondem_true_distribution') and \
                ctx.ondem_true_distribution is not None:
            return ctx.ondem_true_distribution

        if hasattr(dataset, 'dataset'):
            base_dataset = dataset.dataset
        else:
            base_dataset = dataset

        labels = []
        if hasattr(base_dataset, 'targets'):
            labels = base_dataset.targets
        elif hasattr(base_dataset, 'y'):
            labels = base_dataset.y
        if len(labels) == 0:
            return None

        labels = torch.as_tensor(labels, device=ctx.device)
        num_classes = ctx.model.distribution_predictor.net[-1].out_features
        freq = torch.bincount(labels.long(), minlength=num_classes).float()
        freq = freq / freq.sum()
        return freq.detach().cpu()

    def _batch_distribution(self, labels: torch.Tensor, num_classes: int,
                            device: torch.device) -> torch.Tensor:
        labels = labels.view(-1).long()
        freq = torch.bincount(labels, minlength=num_classes).float().to(device)
        freq = freq / freq.sum().clamp_min(1e-12)
        return freq.unsqueeze(0)

    def _extract_gradient_vector(self, ctx) -> torch.Tensor:
        grad_layer_name = self.cfg.ondemfl.grad_layer
        if grad_layer_name:
            modules = dict(ctx.model.task_model.named_modules())
            if grad_layer_name not in modules or \
                    not isinstance(modules[grad_layer_name], nn.Linear):
                raise ValueError(
                    f'Layer `{grad_layer_name}` is not a valid nn.Linear '
                    f'within the backbone; please double check '
                    f'cfg.ondemfl.grad_layer.')
            layer = modules[grad_layer_name]
        else:
            if self._cached_linear is None:
                self._cached_linear = _find_last_linear(ctx.model.task_model)
            layer = self._cached_linear
            if layer is None:
                raise RuntimeError('Unable to locate a linear layer within '
                                   'the backbone to derive gradients.')

        grads = []
        if layer.weight.grad is not None:
            grads.append(layer.weight.grad.view(-1))
        if layer.bias is not None and layer.bias.grad is not None:
            grads.append(layer.bias.grad.view(-1))
        if not grads:
            raise RuntimeError('Gradients for the selected layer are empty.')

        grad_vec = torch.cat(grads).detach()
        return grad_vec


def call_ondemfl_trainer(trainer_type):
    if trainer_type.lower() == 'ondemfl':
        return OnDemFLTrainer
    return None


register_trainer('ondemfl', call_ondemfl_trainer)
logging.getLogger(__name__).info(
    'OnDemFL trainer registered (torch available: %s)', torch is not None)
