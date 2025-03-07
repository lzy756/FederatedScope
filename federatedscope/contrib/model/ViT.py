import torch
from torch import nn
from transformers import ViTForImageClassification, ViTConfig
from federatedscope.register import register_model


class CustomViTForImageClassification(ViTForImageClassification):
    def forward(self, *args, **kwargs):
        # 调用父类的 forward 方法，获取 ImageClassifierOutput
        output = super().forward(*args, **kwargs)
        # 返回 logits（模型输出的未归一化的预测值）
        return output.logits


def load_my_net(model_config, local_data):
    # --- 手动定义ViT模型配置 ---
    config = ViTConfig(
        image_size=224,
        patch_size=16,  # 每个patch的大小 (16x16像素)
        num_channels=3,  # 输入图像通道数（RGB）
        num_labels=model_config.out_channels,  # 类别数
        hidden_size=768,  # 隐藏层维度
        num_hidden_layers=12,  # Transformer编码器层数
        num_attention_heads=12,  # 注意力头数
        intermediate_size=3072,  # FeedForward层中间维度（与原始论文一致）
        hidden_act="gelu",  # 激活函数
        attention_probs_dropout_prob=0.1,
        hidden_dropout_prob=0.1)
    # --- 从零初始化模型（不使用预训练权重）---
    model = CustomViTForImageClassification(config=config)
    return model


def call_my_net(model_config, local_data):
    if model_config.type == "ViT":
        model = load_my_net(model_config, local_data)
        return model


register_model("ViT", call_my_net)
