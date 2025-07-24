import torch
from torchvision.models.resnet import ResNet, BasicBlock


from federatedscope.register import register_model


def model_builder(model_config, input_shape):
    """
    Build the PreActResNet model based on the provided configuration.

    Args:
        model_config: Configuration object containing model parameters.
        input_shape: Shape of the input data (not used in this case).

    Returns:
        An instance of PreActResNet.
    """
    num_classes = model_config.out_channels
    return ResNet(BasicBlock, [1, 1, 1, 1], num_classes=num_classes)


def call_domainnet_resnet(model_config, local_data):
    """
    Call the model builder function to create an instance of PreActResNet.

    Args:
        model_config: Configuration object containing model parameters.
        local_data: Local data (not used in this case).

    Returns:
        An instance of PreActResNet.
    """
    if model_config.type == "domainnet_resnet10":
        return model_builder(model_config, local_data)


# Register the model with the name 'domainnet_resnet10'
register_model("domainnet_resnet10", call_domainnet_resnet)