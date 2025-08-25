import torch
import torch.nn as nn
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


class FDSECNN(nn.Module):
    """FDSE CNN模型，将原始CNN的卷积层替换为分解层"""

    def __init__(self, in_channels=3, num_classes=10, G=2):
        super(FDSECNN, self).__init__()
        
        # 定义原始CNN结构（参考CrossDomainDigitCNN）
        self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        # 将原始卷积层替换为分解层
        self.decomposed_conv1 = DecomposedLayer(self.conv1, G)
        self.decomposed_conv2 = DecomposedLayer(self.conv2, G)
        
        # BN层（可选）
        self.bn1 = nn.BatchNorm2d(16)
        self.bn2 = nn.BatchNorm2d(32)
        
        # 分类器
        self.fc1 = nn.Linear(32 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x, return_dse_outputs=False):
        dse_outputs = {}
        
        # 第一个分解层
        if return_dse_outputs:
            x, dse_out1 = self.decomposed_conv1(x, return_dse_output=True)
            dse_outputs['decomposed_conv1'] = dse_out1
        else:
            x = self.decomposed_conv1(x)
            
        x = self.pool(x)
        
        # 第二个分解层
        if return_dse_outputs:
            x, dse_out2 = self.decomposed_conv2(x, return_dse_output=True)
            dse_outputs['decomposed_conv2'] = dse_out2
        else:
            x = self.decomposed_conv2(x)
            
        x = self.pool(x)
        
        # 分类器
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)  # Output layer

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
            # 获取两个分解层的统计信息
            bn_statistics['decomposed_conv1'] = {
                "running_mean": self.decomposed_conv1.bn_dfe.running_mean.detach()
                .clone()
                .requires_grad_(False),
                "running_var": self.decomposed_conv1.bn_dfe.running_var.detach()
                .clone()
                .requires_grad_(False),
            }
            bn_statistics['decomposed_conv2'] = {
                "running_mean": self.decomposed_conv2.bn_dfe.running_mean.detach()
                .clone()
                .requires_grad_(False),
                "running_var": self.decomposed_conv2.bn_dfe.running_var.detach()
                .clone()
                .requires_grad_(False),
            }

        return bn_statistics


def model_builder(model_config, local_data):
    in_channels = getattr(model_config, "in_channels", 3)
    num_classes = model_config.out_channels
    G = getattr(model_config, "G", 2)
    return FDSECNN(in_channels=in_channels, num_classes=num_classes, G=G)


def call_fdse_cnn(model_config, local_data):
    if model_config.type == "fdse_cnn":
        in_channels = getattr(model_config, "in_channels", 3)
        num_classes = getattr(model_config, "out_channels", 10)
        G = getattr(model_config, "G", 2)
        return model_builder(model_config, local_data)


# Register the model with the name 'fdse_cnn'
register_model("fdse_cnn", call_fdse_cnn)