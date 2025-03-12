import torch
from torch import nn
from transformers import ViTForImageClassification, ViTConfig
from federatedscope.register import register_model


class CustomViTForImageClassification(ViTForImageClassification):
    """适配FederatedScope的ViT派生类，返回原始logits"""
    def forward(self, *args, **kwargs):
        return super().forward(*args, **kwargs).logits


def load_my_net(model_config, local_data):
    """加载预训练ViT并进行目标任务适配"""
    # ---- 核心参数配置 ----
    pretrained_name = "google/vit-base-patch16-224-in21k"  # 与预处理参数严格对齐
    num_labels = model_config.out_channels

    # ---- 加载预训练模型并调整分类头 ----
    model = CustomViTForImageClassification.from_pretrained(
        pretrained_name,
        num_labels=num_labels,
        ignore_mismatched_sizes=True,  # 自动处理分类层维度变化
        cache_dir="./pretrained_models"  # 自定义缓存路径
    )

    # ---- 选择性冻结参数（示例冻结所有非分类层）----
    # if model_config.get("freeze_backbone", True):
    print("[模型配置] 冻结特征提取器，仅训练分类头")
    # 区分特征层和分类头参数
    for param in model.vit.parameters():  # vit部分为特征提取器
        param.requires_grad = False
    for param in model.classifier.parameters():
        param.requires_grad = True

    # 推荐在解冻状态下训练BatchNorm
    def unfreeze_bn_params(module):
        if isinstance(module, nn.BatchNorm2d):
            for param in module.parameters():
                param.requires_grad = True

    model.apply(unfreeze_bn_params)

    # ---- 学习率分层配置（示例）----
    # 注：需要在联邦优化器中配合param_groups实现差异化学习率
    # if model_config.get("differential_lr", False):
    #     model.param_groups = [{
    #         'params': model.vit.parameters(),
    #         'lr': model_config.lr * 0.1
    #     }, {
    #         'params': model.classifier.parameters(),
    #         'lr': model_config.lr
    #     }]

    return model


def call_my_net(model_config, local_data):
    if model_config.type == "ViT":
        return load_my_net(model_config, local_data)


register_model("ViT", call_my_net)
