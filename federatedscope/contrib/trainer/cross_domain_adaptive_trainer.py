"""Cross-domain adaptive trainer that fuses FedLSA's hyperspherical
contrastive learning with the gradient-based distribution predictor used in
On-Demand FL.
"""
from typing import Optional, TYPE_CHECKING
import logging

from federatedscope.core.configs.config import CN as CfgNode

if TYPE_CHECKING:  # pragma: no cover
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
else:  # pragma: no cover
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
    except ImportError:  # pragma: no cover
        torch = None
        nn = None
        F = None

from federatedscope.core.auxiliaries.optimizer_builder import get_optimizer
from federatedscope.core.trainers.context import CtxVar
from federatedscope.core.trainers.enums import MODE, LIFECYCLE
from federatedscope.core.trainers.torch_trainer import GeneralTorchTrainer
from federatedscope.register import register_trainer

logger = logging.getLogger(__name__)


def _require_torch():
    if torch is None or nn is None:
        raise ImportError('PyTorch is required for the cross-domain adaptive '
                          'trainer. Please install torch/torchvision first.')


def _find_last_linear(module: nn.Module) -> Optional[nn.Linear]:
    last_linear = None
    for sub_module in module.modules():
        if isinstance(sub_module, nn.Linear):
            last_linear = sub_module
    return last_linear


class CrossDomainAdaptiveTrainer(GeneralTorchTrainer):
    """Trainer that couples FedLSA losses with the On-Demand FL predictor."""

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

        self.lambda_com = config.fedlsa.lambda_com
        self.tau = config.fedlsa.tau
        self.num_classes = config.model.num_classes
        self.semantic_anchors = None

        self._phase = 'pretrain'
        self._cached_linear: Optional[nn.Linear] = None

        if config.fedlsa.use_projector:
            self._build_hyperspherical_projector()

    # ------------------------------------------------------------------
    # FedLSA-specific utilities
    # ------------------------------------------------------------------
    def _build_hyperspherical_projector(self):
        task_model = getattr(self.ctx.model, 'task_model', self.ctx.model)
        if hasattr(task_model, 'out_channels'):
            in_dim = task_model.out_channels
        elif hasattr(task_model, 'feature_dim'):
            in_dim = task_model.feature_dim
        else:
            in_dim = self.cfg.fedlsa.projector_input_dim
        out_dim = self.cfg.fedlsa.projector_output_dim
        self.projector = nn.Sequential(nn.Linear(in_dim, in_dim),
                                       nn.ReLU(),
                                       nn.Linear(in_dim, out_dim)).to(
                                           self.ctx.device)
        logger.info('Constructed hyperspherical projector: %s -> %s', in_dim,
                    out_dim)

    def update_semantic_anchors(self, anchors):
        self.semantic_anchors = anchors.to(self.ctx.device)

    @staticmethod
    def normalize_embedding(embeddings):
        return F.normalize(embeddings, p=2, dim=1)

    def compute_compactness_loss(self, h_embeddings, labels):
        if self.semantic_anchors is None:
            return torch.tensor(0.0, device=h_embeddings.device)
        logits = torch.matmul(h_embeddings, self.semantic_anchors.T) / self.tau
        return F.cross_entropy(logits, labels)

    def _forward_with_embedding(self, model, x):
        if hasattr(model, 'fc3') and hasattr(model, 'fc2') and hasattr(
                model, 'fc1'):
            z = x
            if hasattr(model, 'conv1'):
                z = model.conv1(z)
                if hasattr(model, 'bn1') and getattr(model, 'use_bn', False):
                    z = model.bn1(z)
                z = model.relu(z)
                z = model.maxpool(z)
            if hasattr(model, 'conv2'):
                z = model.conv2(z)
                if hasattr(model, 'bn2') and getattr(model, 'use_bn', False):
                    z = model.bn2(z)
                z = model.relu(z)
                z = model.maxpool(z)
            z = z.view(z.size(0), -1)
            if hasattr(model, 'dropout'):
                z = F.dropout(z, p=model.dropout, training=model.training)
            embedding = model.relu(model.fc1(z))
            if hasattr(model, 'dropout'):
                z = F.dropout(embedding, p=model.dropout, training=model.training)
            else:
                z = embedding
            z = model.relu(model.fc2(z))
            if hasattr(model, 'dropout'):
                z = F.dropout(z, p=model.dropout, training=model.training)
            pred = model.fc3(z)
            return pred, embedding
        elif hasattr(model, 'fc2') and hasattr(model, 'fc1'):
            z = x
            if hasattr(model, 'conv1'):
                z = model.conv1(z)
                if hasattr(model, 'bn1') and getattr(model, 'use_bn', False):
                    z = model.bn1(z)
                z = model.relu(z)
                z = model.maxpool(z)
            if hasattr(model, 'conv2'):
                z = model.conv2(z)
                if hasattr(model, 'bn2') and getattr(model, 'use_bn', False):
                    z = model.bn2(z)
                z = model.relu(z)
                z = model.maxpool(z)
            z = z.view(z.size(0), -1)
            if hasattr(model, 'dropout'):
                z = F.dropout(z, p=model.dropout, training=model.training)
            embedding = model.relu(model.fc1(z))
            if hasattr(model, 'dropout'):
                z = F.dropout(embedding, p=model.dropout, training=model.training)
            else:
                z = embedding
            pred = model.fc2(z)
            return pred, embedding
        pred = model(x)
        logger.warning('Model lacks explicit embedding heads; using logits '
                       'as embeddings for FedLSA losses.')
        return pred, pred

    # ------------------------------------------------------------------
    # On-Demand FL utilities
    # ------------------------------------------------------------------
    def set_phase(self, phase: str):
        assert phase in ['pretrain', 'ondemand']
        self._phase = phase

    def register_default_hooks_train(self):
        super().register_default_hooks_train()
        self.replace_hook_in_train(self._hook_on_fit_start_init_cross,
                                   target_trigger='on_fit_start',
                                   target_hook_name='_hook_on_fit_start_init')
        self.replace_hook_in_train(self._hook_on_batch_forward_cross,
                                   target_trigger='on_batch_forward',
                                   target_hook_name='_hook_on_batch_forward')
        self.replace_hook_in_train(self._hook_on_batch_backward_cross,
                                   target_trigger='on_batch_backward',
                                   target_hook_name='_hook_on_batch_backward')
        self.register_hook_in_train(self._hook_on_fit_end_cross,
                                    trigger='on_fit_end',
                                    insert_mode='before')
        self.register_hook_in_train(self._hook_on_fit_end_cleanup,
                                    trigger='on_fit_end',
                                    insert_mode='after')

    def _hook_on_fit_start_init_cross(self, ctx):
        _require_torch()
        super()._hook_on_fit_start_init(ctx)
        predictor: nn.Module = ctx.model.distribution_predictor
        dp_cfg: CfgNode = self.cfg.ondemfl.dp_optimizer.clone()
        dp_kwargs = {key: dp_cfg[key] for key in dp_cfg}
        dp_type = dp_kwargs.pop('type')
        dp_lr = dp_kwargs.pop('lr')
        freeze_after_stage1 = self.cfg.ondemfl.freeze_predictor_after_stage1
        if self._phase == 'pretrain' or not freeze_after_stage1:
            ctx.dp_optimizer = get_optimizer(predictor, dp_type, dp_lr,
                                             **dp_kwargs)
        else:
            ctx.dp_optimizer = None

        loss_name = self.cfg.ondemfl.dp_loss.lower()
        if loss_name == 'mse':
            ctx.dp_loss_fn = nn.MSELoss()
        elif loss_name == 'mae':
            ctx.dp_loss_fn = nn.L1Loss()
        elif loss_name in ['kl', 'kldiv']:
            ctx.dp_loss_fn = nn.KLDivLoss(reduction='batchmean')
        else:
            raise ValueError(f'Unsupported dp_loss `{loss_name}`')

        out_dim = predictor.net[-1].out_features
        ctx.dist_pred_sum = CtxVar(torch.zeros(out_dim, device=ctx.device),
                                   LIFECYCLE.ROUTINE)
        ctx.dist_pred_count = CtxVar(0, LIFECYCLE.ROUTINE)
        ctx.last_dp_loss = None

    def _hook_on_batch_forward_cross(self, ctx):
        _require_torch()
        x, label = [_.to(ctx.device) for _ in ctx.data_batch]
        task_model = getattr(ctx.model, 'task_model', ctx.model)
        use_projector = self.cfg.fedlsa.use_projector

        if use_projector:
            if hasattr(task_model, 'get_embedding'):
                z = task_model.get_embedding(x)
                pred = task_model(x)
            else:
                pred, z = self._forward_with_embedding(task_model, x)
            h = self.projector(z)
            h = self.normalize_embedding(h)
        else:
            pred = task_model(x)
            h = None

        if len(label.size()) == 0:
            label = label.unsqueeze(0)

        ctx.y_true = CtxVar(label, LIFECYCLE.BATCH)
        ctx.y_prob = CtxVar(pred, LIFECYCLE.BATCH)
        if h is not None:
            ctx.h_embeddings = CtxVar(h, LIFECYCLE.BATCH)

        loss_ce = ctx.criterion(pred, label)
        if ctx.cur_mode == MODE.TRAIN and h is not None:
            loss_com = self.compute_compactness_loss(h, label)
            ctx.loss_batch = CtxVar(loss_ce + self.lambda_com * loss_com,
                                    LIFECYCLE.BATCH)
            ctx.loss_ce = CtxVar(loss_ce, LIFECYCLE.BATCH)
            ctx.loss_com = CtxVar(loss_com, LIFECYCLE.BATCH)
        else:
            ctx.loss_batch = CtxVar(loss_ce, LIFECYCLE.BATCH)
        ctx.batch_size = CtxVar(len(label), LIFECYCLE.BATCH)
        ctx.loss_task = ctx.loss_batch

    def _hook_on_batch_backward_cross(self, ctx):
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

    def _hook_on_fit_end_cross(self, ctx):
        if ctx.dist_pred_count > 0:
            avg_pred = ctx.dist_pred_sum / ctx.dist_pred_count
            ctx.ondem_pred_distribution = avg_pred.detach().cpu()
        else:
            ctx.ondem_pred_distribution = None
        ctx.ondem_true_distribution = self._compute_true_distribution(ctx)
        if ctx.cur_mode == MODE.TRAIN and hasattr(ctx, 'loss_ce'):
            logger.info('CDA losses - CE: %.4f | COM: %.4f | Predictor loss: %s',
                        ctx.loss_ce.item(), ctx.loss_com.item(),
                        'N/A' if ctx.last_dp_loss is None else
                        f'{ctx.last_dp_loss:.4f}')

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
            except TypeError:  # pragma: no cover
                model.zero_grad()
                for param in model.parameters():
                    if getattr(param, 'grad', None) is not None:
                        param.grad = None
            released = True

        if released and torch.cuda.is_available():
            device = getattr(ctx, 'device', None)
            if isinstance(device, torch.device) and device.type == 'cuda':
                torch.cuda.empty_cache()

    def _compute_true_distribution(self, ctx) -> Optional[torch.Tensor]:
        dataset = ctx.train_loader if ctx.get('train_loader') is not None else \
            ctx.train_data
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
        freq = freq / freq.sum().clamp_min(1e-12)
        return freq.detach().cpu()

    @staticmethod
    def _batch_distribution(labels: torch.Tensor, num_classes: int,
                            device: torch.device) -> torch.Tensor:
        labels = labels.view(-1).long()
        freq = torch.bincount(labels, minlength=num_classes).float().to(device)
        freq = freq / freq.sum().clamp_min(1e-12)
        return freq.unsqueeze(0)

    def _extract_gradient_vector(self, ctx) -> torch.Tensor:
        grad_layer_name = self.cfg.ondemfl.grad_layer
        task_model = ctx.model.task_model
        if grad_layer_name:
            modules = dict(task_model.named_modules())
            if grad_layer_name not in modules or not isinstance(
                    modules[grad_layer_name], nn.Linear):
                raise ValueError(
                    f'Layer `{grad_layer_name}` is not a valid nn.Linear '
                    'within the backbone; please double check '
                    'cfg.ondemfl.grad_layer.')
            layer = modules[grad_layer_name]
        else:
            if self._cached_linear is None:
                self._cached_linear = _find_last_linear(task_model)
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
        return torch.cat(grads).detach()

    # ------------------------------------------------------------------
    # Model sharing helpers
    # ------------------------------------------------------------------
    def get_model_para(self):
        model_para = super().get_model_para()
        model_para = {
            k: v
            for k, v in model_para.items()
            if not k.startswith('distribution_predictor.')
        }
        if hasattr(self, 'projector') and self.cfg.fedlsa.share_projector:
            projector_para = self._param_filter(self.projector.state_dict())
            model_para.update({
                f'projector.{k}': v
                for k, v in projector_para.items()
            })
        return model_para

    def update(self, model_parameters, strict=False):
        model_params = {}
        projector_params = {}
        for key, value in model_parameters.items():
            if key.startswith('projector.'):
                projector_params[key.replace('projector.', '')] = value
            else:
                model_params[key] = value
        if model_params:
            super().update(model_params, strict=strict)
        if projector_params and hasattr(self, 'projector'):
            from federatedscope.core.auxiliaries.utils import (
                merge_param_dict, param2tensor)
            for key in projector_params:
                projector_params[key] = param2tensor(projector_params[key])
            merged_param = merge_param_dict(
                self.projector.state_dict().copy(),
                self._param_filter(projector_params))
            self.projector.load_state_dict(merged_param, strict=strict)


def call_cross_domain_adaptive_trainer(trainer_type: str):
    if trainer_type.lower() == 'cross_domain_adaptive':
        return CrossDomainAdaptiveTrainer
    return None


register_trainer('cross_domain_adaptive', call_cross_domain_adaptive_trainer)
