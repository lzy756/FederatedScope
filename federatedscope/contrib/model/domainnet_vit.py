import torch
import os
from torch import nn
from transformers import ViTForImageClassification, ViTConfig
from federatedscope.register import register_model
import logging

logger = logging.getLogger(__name__)


class CustomViTForImageClassification(ViTForImageClassification):
    """适配FederatedScope的ViT派生类，返回原始logits"""

    def forward(self, *args, **kwargs):
        return super().forward(*args, **kwargs).logits


def adjust_classifier_head(model, num_classes):
    """手动调整分类器头的输出维度"""
    if model.classifier.out_features != num_classes:
        logger.info(f"手动调整分类器: {model.classifier.out_features} -> {num_classes}")
        model.classifier = nn.Linear(
            model.classifier.in_features,
            num_classes,
            bias=model.classifier.bias is not None,
        )
        # 初始化新的分类器权重
        nn.init.normal_(model.classifier.weight, std=0.02)
        if model.classifier.bias is not None:
            nn.init.zeros_(model.classifier.bias)
        logger.info(f"分类器调整完成: {model.classifier}")
    return model


def load_vit(model_config, local_data):
    """加载预训练ViT并进行目标任务适配"""

    # ---- 代理设置 ----
    # 方式1: 从配置文件读取代理设置
    proxy_url = "http://imac.tajuren.cn:34189"
    os.environ["HTTP_PROXY"] = proxy_url
    os.environ["HTTPS_PROXY"] = proxy_url
    logger.info(f"设置代理: {proxy_url}")

    # ---- 核心参数配置 ----
    pretrained_name = "google/vit-base-patch16-224-in21k"  # 与预处理参数严格对齐

    num_labels = model_config.out_channels

    # ---- 加载预训练模型并调整分类头 ----
    logger.info(f"开始加载ViT模型，目标分类数: {num_labels}")
    try:
        model = CustomViTForImageClassification.from_pretrained(
            pretrained_name,
            num_labels=num_labels,
            ignore_mismatched_sizes=True,  # 自动处理分类层维度变化
            cache_dir="./pretrained_models",  # 自定义缓存路径
        )
        logger.info(f"成功加载预训练模型: {pretrained_name}")
        logger.info(f"分类器配置: {model.classifier}")
    except Exception as e:
        logger.error(f"加载预训练模型失败: {e}")
        logger.info("尝试使用镜像站点...")
        # 使用镜像站点作为备选方案
        os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
        model = CustomViTForImageClassification.from_pretrained(
            pretrained_name,
            num_labels=num_labels,
            ignore_mismatched_sizes=True,
            cache_dir="./pretrained_models",
        )
        logger.info(f"通过镜像站点成功加载模型，分类器配置: {model.classifier}")

    # 验证分类器输出维度
    actual_output_dim = model.classifier.out_features
    if actual_output_dim != num_labels:
        model = adjust_classifier_head(model, num_labels)

    # ---- 参数冻结与解冻 ----
    unfreeze_keys = [
        "embeddings.position",  # 位置嵌入
        # "encoder.layer.9",  # 最后3层Transformer块
        "encoder.layer.10",
        "encoder.layer.11",
        "classifier",  # 分类头
    ]
    for name, param in model.named_parameters():
        if any(key in name for key in unfreeze_keys):
            param.requires_grad = True
        else:
            param.requires_grad = False
    # ---- 参数审计 ----
    logger.info("----- 可训练参数审计 -----")
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    classifier_params = sum(p.numel() for p in model.classifier.parameters())

    logger.info(f"模型总参数: {total_params:,}")
    logger.info(
        f"可训练参数: {trainable_params:,} ({trainable_params/total_params:.2%})"
    )
    logger.info(f"分类器参数: {classifier_params:,}")
    logger.info(
        f"分类器配置: 输入维度={model.classifier.in_features}, 输出维度={model.classifier.out_features}"
    )

    return model


def call_vit(model_config, local_data):
    if model_config.type == "domainnet_vit":
        return load_vit(model_config, local_data)


register_model("domainnet_vit", call_vit)
