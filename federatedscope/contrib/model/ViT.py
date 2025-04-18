import torch
from torch import nn
from transformers import ViTForImageClassification, ViTConfig
from federatedscope.register import register_model
import logging

logger = logging.getLogger(__name__)


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

    # ---- 参数冻结与解冻 ----
    unfreeze_keys = [
        "embeddings.position",  # 位置嵌入
        "encoder.layer.9",  # 最后3层Transformer块
        "encoder.layer.10",
        "encoder.layer.11",
        "classifier"  # 分类头
    ]
    for name, param in model.named_parameters():
        if any(key in name for key in unfreeze_keys):
            param.requires_grad = True
        else:
            param.requires_grad = False
    # ---- 参数审计 ----
    logger.info("----- 可训练参数审计 -----")
    trainable_params = sum(p.numel() for p in model.parameters()
                           if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"可训练参数: {trainable_params} / {total_params} "
                f"({trainable_params/total_params:.2%})")

    return model


def call_my_net(model_config, local_data):
    if model_config.type == "ViT":
        return load_my_net(model_config, local_data)


register_model("ViT", call_my_net)
