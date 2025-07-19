import torch.nn as nn
from federatedscope.register import register_model


class CrossDomainDigitCNN(nn.Module):
    def __init__(self, in_channel=3, num_classes=10):
        super(CrossDomainDigitCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, 16, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        # Assuming input size is 32x32, after two pooling layers, the size will be 8x8
        self.fc1 = nn.Linear(32 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool1(x)
        x = self.relu(self.conv2(x))
        x = self.pool2(x)
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)  # Output layer
        return x


def ModelBuilder(in_channel, num_classes):
    """
    Build the CrossDomainDigitCNN model based on the provided configuration.

    Args:
        model_config: Configuration object containing model parameters.
        input_shape: Shape of the input data (not used in this case).

    Returns:
        An instance of CrossDomainDigitCNN.
    """
    return CrossDomainDigitCNN(in_channel=in_channel, num_classes=num_classes)


def call_cross_domain_digit_cnn(model_config, local_data):
    """
    Call the model builder function to create an instance of CrossDomainDigitCNN.

    Args:
        model_config: Configuration object containing model parameters.
        local_data: Local data (not used in this case).

    Returns:
        An instance of CrossDomainDigitCNN.
    """
    in_channel = getattr(model_config, "in_channels", 1)
    num_classes = getattr(model_config, "num_classes", 10)
    return ModelBuilder(in_channel, num_classes)


# Register the model with the name 'cross_domain_digit_cnn'
register_model("cross_domain_digit_cnn", call_cross_domain_digit_cnn)
