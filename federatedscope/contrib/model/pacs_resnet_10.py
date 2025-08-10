import torch
from torchvision.models.resnet import ResNet, BasicBlock


from federatedscope.register import register_model


def model_builder(model_config, input_shape):
    num_classes = model_config.out_channels
    # ResNet with [1, 1, 1, 1] layers and BasicBlock (expansion=1)
    # Feature vector dimension: 512 * BasicBlock.expansion = 512 * 1 = 512
    return ResNet(BasicBlock, [1, 1, 1, 1], num_classes=num_classes)


def call_pacs_resnet(model_config, local_data):
    if model_config.type == "pacs_resnet10":
        return model_builder(model_config, local_data)


# Register the model with the name 'pacs_resnet10'
register_model("pacs_resnet10", call_pacs_resnet)
