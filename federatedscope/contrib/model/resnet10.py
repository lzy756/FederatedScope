import torch
import torch.nn as nn
from torch.nn.functional import relu, avg_pool2d
from typing import List
from federatedscope.register import register_model

import logging

logger = logging.getLogger(__name__)


def conv3x3(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """
    Instantiates a 3x3 convolutional layer with no bias.
    :param in_planes: number of input channels
    :param out_planes: number of output channels
    :param stride: stride of the convolution
    :return: convolutional layer
    """
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    """
    The basic block of ResNet.
    """
    expansion = 1

    def __init__(self, in_planes: int, planes: int, stride: int = 1) -> None:
        """
        Instantiates the basic block of the network.
        :param in_planes: the number of input channels
        :param planes: the number of channels (to be possibly expanded)
        """
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute a forward pass.
        :param x: input tensor (batch_size, input_size)
        :return: output tensor (10)
        """
        out = relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = relu(out)
        return out


class ResNet10(nn.Module):
    """
    ResNet10 network architecture. Designed for complex datasets.
    Based on RethinkFL implementation.
    """

    def __init__(self, block: BasicBlock, num_blocks: List[int],
                 num_classes: int, nf: int = 64, in_channels: int = 3) -> None:
        """
        Instantiates the layers of the network.
        :param block: the basic ResNet block
        :param num_blocks: the number of blocks per layer
        :param num_classes: the number of output classes
        :param nf: the number of filters
        :param in_channels: number of input channels
        """
        super(ResNet10, self).__init__()
        self.name = 'resnet10'
        self.in_planes = nf
        self.block = block
        self.num_classes = num_classes
        self.nf = nf

        self.conv1 = conv3x3(in_channels, nf * 1)
        self.bn1 = nn.BatchNorm2d(nf * 1)
        self.layer1 = self._make_layer(block, nf * 1, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, nf * 2, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, nf * 4, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, nf * 8, num_blocks[3], stride=2)
        self.linear = nn.Linear(nf * 8 * block.expansion, num_classes)

        self._features = nn.Sequential(self.conv1,
                                       self.bn1,
                                       nn.ReLU(),
                                       self.layer1,
                                       self.layer2,
                                       self.layer3,
                                       self.layer4)
        self.cls = self.linear

        self.encoder = nn.Sequential(
            nn.Linear(nf * 8 * block.expansion, nf * 8 * block.expansion),
            nn.ReLU(inplace=True),
            nn.Linear(nf * 8 * block.expansion, 512)
        )

    def _make_layer(self, block: BasicBlock, planes: int,
                    num_blocks: int, stride: int) -> nn.Module:
        """
        Instantiates a ResNet layer.
        :param block: ResNet basic block
        :param planes: channels across the network
        :param num_blocks: number of blocks
        :param stride: stride
        :return: ResNet layer
        """
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features from input."""
        out = self._features(x)
        out = avg_pool2d(out, out.shape[2])
        feat = out.view(out.size(0), -1)
        return feat

    def encoders(self, x: torch.Tensor) -> torch.Tensor:
        """Extract encoded features from input."""
        out = self._features(x)
        out = avg_pool2d(out, out.shape[2])
        feat = out.view(out.size(0), -1)
        feat = self.encoder(feat)
        return feat

    def classifier(self, x: torch.Tensor) -> torch.Tensor:
        """Apply classifier to features."""
        out = self.cls(x)
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        """
        out = relu(self.bn1(self.conv1(x)))  # 64, 32, 32
        if hasattr(self, 'maxpool'):
            out = self.maxpool(out)
        out = self.layer1(out)  # -> 64, 32, 32
        out = self.layer2(out)  # -> 128, 16, 16
        out = self.layer3(out)  # -> 256, 8, 8
        out = self.layer4(out)  # -> 512, 4, 4
        out = avg_pool2d(out, out.shape[2])  # -> 512, 1, 1
        feature = out.view(out.size(0), -1)  # 512
        out = self.cls(feature)
        return out


def create_resnet10(num_classes: int, nf: int = 64, in_channels: int = 3) -> ResNet10:
    """
    Create a ResNet10 network instance.
    :param num_classes: number of output classes
    :param nf: number of filters
    :param in_channels: number of input channels
    :return: ResNet10 network
    """
    return ResNet10(BasicBlock, [1, 1, 1, 1], num_classes, nf, in_channels)


def resnet10_builder(model_config, input_shape):
    """
    Build ResNet10 model based on configuration.

    Args:
        model_config: Configuration object containing model parameters
        input_shape: Input shape tuple (channels, height, width) or just (features,)

    Returns:
        ResNet10 model instance
    """
    # Extract configuration parameters
    num_classes = getattr(model_config, 'out_channels', 10)
    nf = getattr(model_config, 'hidden', 64)  # number of filters

    # Determine input channels from input_shape
    if input_shape is not None:
        if isinstance(input_shape, (list, tuple)) and len(input_shape) >= 3:
            # For image data: (channels, height, width)
            in_channels = input_shape[0] if len(input_shape) == 3 else input_shape[-3]
        else:
            # Default to 3 channels for RGB images
            in_channels = 3
    else:
        in_channels = getattr(model_config, 'in_channels', 3)

    logger.info(f"Building ResNet10 with {num_classes} classes, {nf} filters, {in_channels} input channels")

    model = create_resnet10(num_classes=num_classes, nf=nf, in_channels=in_channels)

    return model


def call_resnet10(model_config, input_shape=None):
    """
    Entry point for ResNet10 model registration.

    Args:
        model_config: Configuration object
        input_shape: Input shape information

    Returns:
        ResNet10 model if type matches, None otherwise
    """
    if model_config.type.lower() == "resnet10":
        return resnet10_builder(model_config, input_shape)
    return None


# Register the ResNet10 model
register_model("resnet10", call_resnet10)

def main():
    """
    主函数：创建模型并打印所有参数名称
    """
    model = create_resnet10(num_classes=10, nf=64, in_channels=3)
    for name, param in model.named_parameters():
        print(f"  - {name}")

if __name__ == "__main__":
    main()