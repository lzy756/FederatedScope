import torch
import torch.nn as nn
from torchvision.models import AlexNet
from federatedscope.register import register_model


class DFE(nn.Module):
    """领域无关特征提取器 (Domain-Free Extractor)"""

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super().__init__()
        # 仅包含卷积层（无BN，BN将在DSE之后使用）
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False,  # 偏置由后续BN层处理
        )

    def forward(self, x):
        return self.conv(x)  # 仅输出卷积结果（无BN/激活）


class DSE(nn.Module):
    """领域特定偏移消除器 (Domain-Specific Eraser)"""

    def __init__(self, num_dfe_channels, G):
        super().__init__()
        self.G = G  # 每个DFE通道扩展为G个通道
        # 为每个DFE输出通道创建独立的DSE子模块
        self.submodules = nn.ModuleList(
            [
                nn.Sequential(
                    nn.BatchNorm2d(1),  # 单通道BN（公式3中的BN_DSE）
                    nn.ReLU(inplace=False),
                    nn.Conv2d(
                        in_channels=1,
                        out_channels=G,
                        kernel_size=1,  # 1x1卷积实现线性扩展
                        stride=1,
                        padding=0,
                        bias=False,
                    ),
                )
                for _ in range(num_dfe_channels)
            ]
        )

    def forward(self, x):
        # x: DFE输出，形状为(B, T_dfe, H, W)
        dse_outputs = []
        # 按通道拆分并分别处理
        for i in range(x.size(1)):
            # 提取单通道特征 (B, 1, H, W)
            single_channel = x[:, i : i + 1, :, :]
            # 经过DSE子模块处理
            dse_out = self.submodules[i](single_channel)
            dse_outputs.append(dse_out)
        # 拼接所有输出通道，恢复为原始通道数 (B, T_dfe*G, H, W)
        return torch.cat(dse_outputs, dim=1)


class DecomposedLayer(nn.Module):
    """完整的分解层：DFE + DSE + DFE的BN层"""

    def __init__(self, original_conv, G=2):
        super().__init__()
        T_original = original_conv.out_channels
        self.T_dfe = (T_original + G - 1) // G

        # 初始化DFE模块
        self.dfe = DFE(
            in_channels=original_conv.in_channels,
            out_channels=self.T_dfe,
            kernel_size=original_conv.kernel_size,
            stride=original_conv.stride,
            padding=original_conv.padding,
        )

        # 初始化DSE模块
        self.dse = DSE(num_dfe_channels=self.T_dfe, G=G)

        # DFE的BN层（放置在DSE之后，公式3中的BN_DFE）
        self.bn_dfe = nn.BatchNorm2d(T_original)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x, return_dse_output=False):
        # 1. DFE提取领域无关特征（无BN）
        x_dfe = self.dfe(x)

        # 2. DSE处理领域偏移（含单通道BN）
        x_dse = self.dse(x_dfe)

        # 3. 应用DFE的BN层和激活函数
        if return_dse_output:
            # 如果需要返回DSE输出，先克隆一份用于保存
            x_dse_saved = x_dse.clone()  # 克隆避免计算图冲突
            x_out = self.relu(self.bn_dfe(x_dse))
            return x_out, x_dse_saved
        else:
            x_out = self.relu(self.bn_dfe(x_dse))
            return x_out


class FDSEModel(nn.Module):
    """完整的FDSE模型，将原始模型的卷积层替换为分解层"""

    def __init__(self, original_model, G=2):
        super().__init__()
        self.features = self._decompose_features(original_model.features, G)
        self.classifier = original_model.classifier  # 保留原始分类器

    def _decompose_features(self, original_features, G):
        """替换特征提取部分的卷积层为分解层"""
        decomposed = nn.Sequential()
        for i, layer in enumerate(original_features):
            if isinstance(layer, nn.Conv2d):
                # 替换卷积层为分解层
                decomposed.add_module(
                    f"decomposed_layer_{i}", DecomposedLayer(layer, G)
                )
            elif isinstance(layer, nn.ReLU):
                # 替换ReLU层，设置inplace=False避免计算图冲突
                decomposed.add_module(f"layer_{i}", nn.ReLU(inplace=False))
            else:
                # 保留池化、BN等其他层
                decomposed.add_module(f"layer_{i}", layer)
        return decomposed

    def forward(self, x, return_dse_outputs=False):
        dse_outputs = {}

        for name, layer in self.features.named_children():
            if name.startswith("decomposed_layer_"):
                # 对于分解层，获取DSE输出
                if return_dse_outputs:
                    x, dse_out = layer(x, return_dse_output=True)
                    dse_outputs[name] = dse_out
                else:
                    x = layer(x)
            else:
                # 对于其他层（池化、BN等），直接传递
                x = layer(x)

        x = x.view(x.size(0), -1)  # 展平特征图
        x = self.classifier(x)

        if return_dse_outputs:
            return x, dse_outputs
        else:
            return x

    def get_bn_dfe_statistics(self):
        """
        获取所有DecomposedLayer中bn_dfe的统计信息（均值和方差）

        Returns:
            dict: 包含每个DecomposedLayer的bn_dfe统计信息的字典
                - 'layer_name': {
                    'mean': tensor,       # 运行时均值
                    'var': tensor,        # 运行时方差
                  }
        """
        with torch.no_grad():
            bn_statistics = {}
            # decomposed_layer_0
            for name, layer in self.features.named_children():
                if name.startswith("decomposed_layer_"):
                    bn_dfe = layer.bn_dfe
                    bn_statistics[name] = {
                        "running_mean": bn_dfe.running_mean.detach()
                        .clone()
                        .requires_grad_(False),
                        "running_var": bn_dfe.running_var.detach()
                        .clone()
                        .requires_grad_(False),
                    }

        return bn_statistics


def model_builder(model_config, local_data):
    num_classes = model_config.out_channels
    drop_out = 0.5
    alexnet_model = AlexNet(num_classes=num_classes, dropout=drop_out)
    return FDSEModel(alexnet_model, G=2)


def call_fdse_alexnet(model_config, local_data):
    if model_config.type == "fdse_alexnet":
        return model_builder(model_config, local_data)


register_model("fdse_alexnet", call_fdse_alexnet)
