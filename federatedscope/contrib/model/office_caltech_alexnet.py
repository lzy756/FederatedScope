import torch
from torchvision.models import AlexNet


from federatedscope.register import register_model


def model_builder(model_config, input_shape):
    num_classes = model_config.out_channels
    # AlexNet with configurable dropout rate
    # Feature vector dimension: 4096 (before the final classifier layer)
    dropout = getattr(model_config, "dropout", 0.5)  # Default dropout 0.5
    return AlexNet(num_classes=num_classes, dropout=dropout)


def call_office_caltech_alexnet(model_config, local_data):
    if model_config.type == "office_caltech_alexnet":
        return model_builder(model_config, local_data)


# Register the model with the name 'office_caltech_alexnet'
register_model("office_caltech_alexnet", call_office_caltech_alexnet)
