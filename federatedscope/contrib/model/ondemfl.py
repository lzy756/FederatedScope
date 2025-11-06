import copy
from typing import Iterable, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from yacs.config import CfgNode as CN

from federatedscope.register import register_model


class DistributionPredictor(nn.Module):
    """
    Simple multilayer perceptron used to approximate the class distribution
    of a client given the flattened gradient of the backbone's last layer.
    """
    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 hidden_sizes: Optional[Iterable[int]] = None,
                 temperature: float = 1.0):
        super().__init__()
        if hidden_sizes is None:
            hidden_sizes = []
        layers: List[nn.Module] = []
        prev_dim = input_dim
        for hidden_dim in hidden_sizes:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, output_dim))
        self.net = nn.Sequential(*layers)
        self.temperature = temperature

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.net(x)
        if self.temperature != 1.0:
            logits = logits / self.temperature
        return F.softmax(logits, dim=-1)


class OnDemFLModel(nn.Module):
    """
    Composite model that bundles the backbone (task) model together with the
    class-distribution predictor.
    """
    def __init__(self, task_model: nn.Module, predictor: nn.Module):
        super().__init__()
        self.task_model = task_model
        self.distribution_predictor = predictor

    def forward(self, *args, **kwargs):
        return self.task_model(*args, **kwargs)


def _get_last_linear_layer(module: nn.Module) -> Optional[nn.Linear]:
    last_linear = None
    for sub_module in module.modules():
        if isinstance(sub_module, nn.Linear):
            last_linear = sub_module
    return last_linear


def _get_predictor_input_dim(task_model: nn.Module,
                             explicit_layer_name: str = '') -> int:
    if explicit_layer_name:
        module = dict(task_model.named_modules()).get(explicit_layer_name,
                                                      None)
        if module is None or not isinstance(module, nn.Linear):
            raise ValueError(
                f'Layer "{explicit_layer_name}" is not a valid nn.Linear '
                f'within the task model.')
        weight_dim = module.weight.numel()
        bias_dim = module.bias.numel() if module.bias is not None else 0
        return weight_dim + bias_dim

    last_linear = _get_last_linear_layer(task_model)
    if last_linear is None:
        raise ValueError('Failed to locate the last nn.Linear layer within '
                         'the task model; please set '
                         '`cfg.ondemfl.grad_layer` explicitly.')
    weight_dim = last_linear.weight.numel()
    bias_dim = last_linear.bias.numel() if last_linear.bias is not None else 0
    return weight_dim + bias_dim


def call_ondemfl_model(model_config, input_shape=None):
    if model_config.type.lower() != 'ondemfl':
        return None

    if not hasattr(model_config, 'backbone'):
        raise ValueError('`model.backbone` must be provided when '
                         'model.type == "ondemfl"')

    backbone_cfg = CN(new_allowed=True)
    backbone_cfg.type = model_config.backbone
    if hasattr(model_config, 'backbone_config'):
        cfg_to_merge = model_config.backbone_config
        if isinstance(cfg_to_merge, dict):
            cfg_to_merge = CN(cfg_to_merge)
        backbone_cfg.merge_from_other_cfg(cfg_to_merge)

    if not hasattr(backbone_cfg, 'out_channels') and hasattr(
            model_config, 'out_channels'):
        backbone_cfg.out_channels = model_config.out_channels

    if input_shape is not None:
        input_shape = tuple(input_shape)
        backbone_cfg = backbone_cfg.clone()
        backbone_cfg.input_shape = input_shape

    from federatedscope.core.auxiliaries.model_builder import get_model
    task_model = get_model(backbone_cfg, None)
    if task_model is None:
        raise ValueError(
            f'Failed to build backbone model `{model_config.backbone}`.')

    if not hasattr(model_config, 'out_channels'):
        raise ValueError('`model.out_channels` must be specified for OnDemFL '
                         'to define the predictor output dimension.')

    layer_name = getattr(model_config, 'grad_layer', '')
    predictor_input_dim = _get_predictor_input_dim(task_model, layer_name)

    hidden_sizes = getattr(model_config, 'distribution_hidden', None)
    if hidden_sizes is None:
        hidden_sizes = [
            max(model_config.out_channels * 8, predictor_input_dim // 2),
            max(model_config.out_channels * 4, predictor_input_dim // 4)
        ]
    temperature = getattr(model_config, 'distribution_temperature', 1.0)
    predictor = DistributionPredictor(predictor_input_dim,
                                      model_config.out_channels,
                                      hidden_sizes=hidden_sizes,
                                      temperature=temperature)

    return OnDemFLModel(task_model, predictor)


register_model('ondemfl', call_ondemfl_model)
