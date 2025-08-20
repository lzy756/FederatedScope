import torch
import torch.nn as nn

from federatedscope.register import register_model
from torchvision.models.resnet import ResNet, BasicBlock


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
                    # 替换为深度卷积：groups=1（与输入通道数一致），输出通道数=G
                    nn.Conv2d(
                        in_channels=1,
                        out_channels=G,
                        kernel_size=(3, 3),
                        stride=1,
                        padding=1,
                        groups=1,
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


class DecomposedBasicBlock(nn.Module):
    """分解后的BasicBlock"""

    def __init__(self, original_block, G):
        super().__init__()

        # 分解conv1和conv2
        self.conv1 = DecomposedLayer(original_block.conv1, G)
        self.bn1 = original_block.bn1
        self.relu = nn.ReLU(inplace=False)

        self.conv2 = DecomposedLayer(original_block.conv2, G)
        self.bn2 = original_block.bn2

        # 处理downsample（如果存在）- 不分解downsample层
        if original_block.downsample is not None:
            self.downsample = original_block.downsample
        else:
            self.downsample = None

        self.stride = original_block.stride

    def forward(self, x, return_dse_outputs=False):
        dse_outputs = {}
        identity = x

        # 主路径
        if return_dse_outputs:
            out, dse_out1 = self.conv1(x, return_dse_output=True)
            dse_outputs["conv1"] = dse_out1
        else:
            out = self.conv1(x)

        out = self.bn1(out)
        out = self.relu(out)

        if return_dse_outputs:
            out, dse_out2 = self.conv2(out, return_dse_output=True)
            dse_outputs["conv2"] = dse_out2
        else:
            out = self.conv2(out)

        out = self.bn2(out)

        # 跳跃连接
        if self.downsample is not None:
            # downsample层不进行分解，直接使用原始层
            identity = self.downsample(x)
        else:
            identity = x

        out += identity
        out = self.relu(out)

        if return_dse_outputs:
            return out, dse_outputs
        else:
            return out


class FDSEModel(nn.Module):
    """完整的FDSE模型，将ResNet模型的卷积层替换为分解层"""

    def __init__(self, original_model, G=2):
        super().__init__()
        self.G = G

        # 复制并分解ResNet的各个组件
        self.conv1 = self._decompose_conv_layer(original_model.conv1, G)
        self.bn1 = original_model.bn1
        self.relu = nn.ReLU(inplace=False)  # 避免inplace操作
        self.maxpool = original_model.maxpool

        # 分解各个残差块层
        self.layer1 = self._decompose_layer_block(original_model.layer1, G)
        self.layer2 = self._decompose_layer_block(original_model.layer2, G)
        self.layer3 = self._decompose_layer_block(original_model.layer3, G)
        self.layer4 = self._decompose_layer_block(original_model.layer4, G)

        self.avgpool = original_model.avgpool
        self.fc = original_model.fc

    def _decompose_conv_layer(self, conv_layer, G):
        """分解单个卷积层"""
        if isinstance(conv_layer, nn.Conv2d):
            return DecomposedLayer(conv_layer, G)
        else:
            return conv_layer

    def _decompose_layer_block(self, layer_block, G):
        """分解ResNet的layer块（包含多个BasicBlock）"""
        decomposed_block = nn.Sequential()

        for i, block in enumerate(layer_block):
            if hasattr(block, "conv1") and hasattr(block, "conv2"):
                # 这是一个BasicBlock，需要分解其中的卷积层
                decomposed_basic_block = self._decompose_basic_block(block, G)
                decomposed_block.add_module(str(i), decomposed_basic_block)
            else:
                # 保持原样
                decomposed_block.add_module(str(i), block)

        return decomposed_block

    def _decompose_basic_block(self, basic_block, G):
        """分解BasicBlock中的卷积层"""
        return DecomposedBasicBlock(basic_block, G)

    def forward(self, x, return_dse_outputs=False):
        all_dse_outputs = {}

        # conv1
        if return_dse_outputs:
            x, dse_out = self.conv1(x, return_dse_output=True)
            all_dse_outputs["conv1"] = dse_out
        else:
            x = self.conv1(x)

        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # layer1-4
        for layer_name, layer_block in [
            ("layer1", self.layer1),
            ("layer2", self.layer2),
            ("layer3", self.layer3),
            ("layer4", self.layer4),
        ]:
            for i, block in enumerate(layer_block):
                if hasattr(block, "conv1"):  # 分解后的BasicBlock
                    if return_dse_outputs:
                        x, block_dse_outputs = block(x, return_dse_outputs=True)
                        # 为每个block的DSE输出添加层级前缀
                        for key, value in block_dse_outputs.items():
                            all_dse_outputs[f"{layer_name}.{i}.{key}"] = value
                    else:
                        x = block(x)
                else:
                    x = block(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        if return_dse_outputs:
            return x, all_dse_outputs
        else:
            return x

    def get_bn_dfe_statistics(self):
        """
        获取所有DecomposedLayer中bn_dfe的统计信息（均值和方差）

        Returns:
            dict: 包含每个DecomposedLayer的bn_dfe统计信息的字典
                - 'layer_name': {
                    'running_mean': tensor,       # 运行时均值
                    'running_var': tensor,        # 运行时方差
                  }
        """
        with torch.no_grad():
            bn_statistics = {}

            # conv1的BN统计
            if isinstance(self.conv1, DecomposedLayer):
                bn_dfe = self.conv1.bn_dfe
                bn_statistics["conv1"] = {
                    "running_mean": bn_dfe.running_mean.detach()
                    .clone()
                    .requires_grad_(False),
                    "running_var": bn_dfe.running_var.detach()
                    .clone()
                    .requires_grad_(False),
                }

            # 各layer块的BN统计
            for layer_name, layer_block in [
                ("layer1", self.layer1),
                ("layer2", self.layer2),
                ("layer3", self.layer3),
                ("layer4", self.layer4),
            ]:
                for i, block in enumerate(layer_block):
                    if hasattr(block, "conv1") and isinstance(
                        block.conv1, DecomposedLayer
                    ):
                        # conv1的BN统计
                        bn_dfe = block.conv1.bn_dfe
                        bn_statistics[f"{layer_name}.{i}.conv1"] = {
                            "running_mean": bn_dfe.running_mean.detach()
                            .clone()
                            .requires_grad_(False),
                            "running_var": bn_dfe.running_var.detach()
                            .clone()
                            .requires_grad_(False),
                        }

                    if hasattr(block, "conv2") and isinstance(
                        block.conv2, DecomposedLayer
                    ):
                        # conv2的BN统计
                        bn_dfe = block.conv2.bn_dfe
                        bn_statistics[f"{layer_name}.{i}.conv2"] = {
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
    resnet10_model = ResNet(BasicBlock, [1, 1, 1, 1], num_classes=num_classes)
    return FDSEModel(resnet10_model, G=8)


def call_fdse_alexnet(model_config, local_data):
    if model_config.type == "fdse_resnet10":
        return model_builder(model_config, local_data)


register_model("fdse_resnet10", call_fdse_alexnet)
