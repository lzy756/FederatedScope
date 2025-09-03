import torch
from torch import nn
from transformers import CLIPVisionModel, CLIPModel
from federatedscope.register import register_model
import logging

logger = logging.getLogger(__name__)


def adapter_residual_hook(module, input, output):
    """
    把 module._adapter 作为残差注入到该层的 hidden_states 上。
    兼容 4.11.x 下 CLIPVisionTransformer 层输出的几种形态：
    - 直接张量
    - 元组 (hidden_states, ...)
    - 含 hidden_states 属性的对象
    """
    adp = getattr(module, "_adapter", None)
    if adp is None:
        return output

    # 取出 hidden states
    if isinstance(output, tuple):
        hs = output[0]
        hs = adp(hs)
        return (hs, ) + output[1:]
    elif hasattr(output, "hidden_states"):
        hs = output.hidden_states
        hs = adp(hs)
        output.hidden_states = hs
        return output
    else:
        hs = output
        hs = adp(hs)
        return hs


class Adapter(nn.Module):
    """最小瓶颈 Adapter：x + W2(GELU(W1(x))) 残差"""
    def __init__(self, dim: int, bottleneck: int = 64):
        super().__init__()
        self.down = nn.Linear(dim, bottleneck, bias=False)
        self.act = nn.GELU()
        self.up = nn.Linear(bottleneck, dim, bias=False)

    def forward(self, x):
        return x + self.up(self.act(self.down(x)))


class CLIPVisionWithHead(nn.Module):
    """
    封装 CLIP ViT-B/16 视觉骨干，支持：
      - 线性探测（仅 classifier 训练）
      - 最后 K 层注入 Adapter（训练 adapter+classifier）
    返回 logits，且为 FedSAK 标记共享参数：p._is_shared = True
    """
    def __init__(
        self,
        num_labels: int,
        pretrained_name: str = "pretrained_models/clip-vit-base-patch16",
        cache_dir: str = "./pretrained_models",
        strategy: str = "linear",  # "linear" or "adapter"
        adapter_last_k: int = 3,  # 最后 K 层加 Adapter
        adapter_bottleneck: int = 64,  # Adapter 瓶颈维度
        share: str = "adapter",  # "head" | "adapter" | "head+adapter"
        unfreeze_ln: bool = True  # 是否解冻 LayerNorm（但默认不共享）
    ):
        super().__init__()
        # self.vision = CLIPVisionModel.from_pretrained(
        #     pretrained_name,
        #     local_files_only=True
        # )
        clip = CLIPModel.from_pretrained(pretrained_name,
                                         local_files_only=True)
        self.vision = clip.vision_model
        # self.processor = CLIPImageProcessor.from_pretrained(
        #     pretrained_name,
        #     cache_dir=cache_dir
        # )
        hidden = self.vision.config.hidden_size  # 768 for ViT-B/16
        self.classifier = nn.Linear(hidden, num_labels)

        # ---- 冻结/解冻策略 ----
        for p in self.vision.parameters():
            p.requires_grad = False

        # 仅线性探测：只训练 classifier
        if strategy == "linear":
            for p in self.classifier.parameters():
                p.requires_grad = True

            # （可选）解冻 LN 提升自适应性，但不共享
            if unfreeze_ln:
                for n, p in self.vision.named_parameters():
                    if "layer_norm" in n or "ln_" in n or "norm" in n:
                        p.requires_grad = True

            self.adapters = None

        # Adapter：在最后 K 层插 Adapter，训练 adapter + classifier
        elif strategy == "adapter":
            # 准备最后 K 层
            layers = self.vision.encoder.layers
            assert 1 <= adapter_last_k <= len(layers)
            target_idx = list(range(len(layers) - adapter_last_k, len(layers)))

            self.adapters = nn.ModuleDict()
            self._hooks = []
            for i in target_idx:
                adp = Adapter(hidden, adapter_bottleneck)
                self.adapters[f"layer_{i}"] = adp

                # 把 adapter 挂到该层上，避免闭包
                setattr(layers[i], "_adapter", adp)

                # 用“顶层函数”注册 hook（可被 pickle）
                h = layers[i].register_forward_hook(adapter_residual_hook)
                self._hooks.append(h)

            # 训练 adapter + classifier
            for p in self.classifier.parameters():
                p.requires_grad = True
            for p in self.adapters.parameters():
                p.requires_grad = True

            if unfreeze_ln:
                for n, p in self.vision.named_parameters():
                    if "layer_norm" in n or "ln_" in n or "norm" in n:
                        p.requires_grad = True
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        # 审计
        trainable = sum(p.numel() for p in self.parameters()
                        if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        logger.info(f"[CLIP_ViT] Trainable params: {trainable} / {total} "
                    f"({trainable/total:.2%})")

    def forward(self, pixel_values=None, **kwargs):
        """
        期望上游 dataloader 已将图像预处理为 CLIP 的 pixel_values: [B, 3, 224, 224]
        """
        outputs = self.vision(pixel_values=pixel_values,
                              output_hidden_states=False)
        # CLS token 在最后一个 patch 的 pooled_output 中
        pooled = outputs.pooler_output  # [B, hidden]
        logits = self.classifier(pooled)
        return logits


# -------- FederatedScope 接口 --------


def load_my_clip_vit(model_config, local_data):
    num_labels = model_config.out_channels

    # 可从 model_config 读取自定义开关
    strategy = getattr(model_config, "strategy",
                       "adapter")  # "linear" / "adapter"
    adapter_last_k = int(getattr(model_config, "adapter_last_k", 3))
    adapter_bottleneck = int(getattr(model_config, "adapter_bottleneck", 64))
    share = getattr(model_config, "share",
                    "adapter")  # "head"|"adapter"|"head+adapter"
    unfreeze_ln = bool(getattr(model_config, "unfreeze_ln", True))

    model = CLIPVisionWithHead(
        num_labels=num_labels,
        pretrained_name="pretrained_models/clip-vit-base-patch16",
        cache_dir="./pretrained_models",
        strategy=strategy,
        adapter_last_k=adapter_last_k,
        adapter_bottleneck=adapter_bottleneck,
        share=share,
        unfreeze_ln=unfreeze_ln)
    return model


def call_my_net(model_config, local_data):
    if model_config.type == "CLIP_ViT":  # 你也可以把旧标识 ViT 映射到新加载器
        return load_my_clip_vit(model_config, local_data)


register_model("CLIP_ViT", call_my_net)
