import torch
import torch.nn as nn
from torch.autograd import Function

from federatedscope.register import register_model


class ToyFeatureExtractor(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()
        self.conv0 = nn.Conv2d(in_channels, 32, kernel_size=5, padding=2)
        self.conv1 = nn.Conv2d(32, 48, kernel_size=5, padding=2)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = torch.relu(self.conv0(x))
        x = self.pool(x)
        x = torch.relu(self.conv1(x))
        x = self.pool(x)
        return torch.flatten(x, 1)


class LightHead(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.fc1 = nn.Linear(in_features, 100)
        self.dropout = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(100, out_features)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)


class DomainHead(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.fc1 = nn.Linear(in_features, 100)
        self.dropout = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(100, 2)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)


class _GradReverseFunc(Function):
    @staticmethod
    def forward(ctx, input_tensor, lambd):
        ctx.lambd = lambd
        return input_tensor.view_as(input_tensor)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.lambd, None


def grad_reverse(x, lambd):
    return _GradReverseFunc.apply(x, lambd)


class FedMMDANN(nn.Module):
    def __init__(self, in_channels=3, num_classes=10):
        super().__init__()
        self.feature = ToyFeatureExtractor(in_channels)
        feature_dim = 7 * 7 * 48
        self.task_head = LightHead(feature_dim, num_classes)
        self.domain_head = DomainHead(feature_dim)
        self.grl_lambda = 1.0

    def set_grl_lambda(self, lambd):
        self.grl_lambda = lambd

    def forward(self, x, return_domain=False):
        if x.dtype != torch.float32:
            x = x.float()
        mean = x.mean()
        x = (x - mean) / 255.
        feat = self.feature(x)
        logits = self.task_head(feat)
        domain_feat = grad_reverse(feat, self.grl_lambda)
        domain_logits = self.domain_head(domain_feat)
        if return_domain:
            return logits, domain_logits
        return logits


def call_fedmm_model(model_config, local_data):
    if model_config.type.lower() != 'fedmm_dann':
        return None
    in_channels = model_config.in_channels or 3
    num_classes = model_config.out_channels or 10
    model = FedMMDANN(in_channels=in_channels, num_classes=num_classes)
    return model


register_model('fedmm_dann', call_fedmm_model)
