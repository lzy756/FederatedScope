import torch
import torch.nn as nn
import torch.nn.functional as F
from federatedscope.register import register_model


class SVHNCNN(nn.Module):
    """
    CNN Model for SVHN Dataset based on the svhn_model.ipynb notebook
    
    Architecture:
    INPUT -> [CONV -> RELU -> CONV -> RELU -> POOL] -> DROPOUT -> [FC -> RELU] -> FC
    
    Original TensorFlow architecture from notebook:
    - Conv2d(32 filters, 5x5 kernel, same padding, ReLU)
    - MaxPool2d(2x2, stride=2)
    - Conv2d(64 filters, 5x5 kernel, same padding, ReLU)  
    - MaxPool2d(2x2, stride=2)
    - Dense(256 units, ReLU)
    - Dropout(rate=discard_rate)
    - Dense(10 units) - output layer
    
    Input: 32x32x1 grayscale images
    Output: 10 classes (digits 0-9)
    """
    
    def __init__(self, in_channels=1, num_classes=10, hidden_dim=256, dropout_rate=0.7):
        super(SVHNCNN, self).__init__()
        
        # Convolutional Layer #1: 32 filters, 5x5 kernel, same padding
        self.conv1 = nn.Conv2d(in_channels=in_channels, 
                              out_channels=32, 
                              kernel_size=5, 
                              padding=2)  # padding=2 for 'same' padding with 5x5 kernel
        
        # Pooling Layer #1: 2x2 max pooling, stride=2
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Convolutional Layer #2: 64 filters, 5x5 kernel, same padding  
        self.conv2 = nn.Conv2d(in_channels=32, 
                              out_channels=64, 
                              kernel_size=5, 
                              padding=2)  # padding=2 for 'same' padding with 5x5 kernel
        
        # Pooling Layer #2: 2x2 max pooling, stride=2
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # After two pooling layers: 32x32 -> 16x16 -> 8x8
        # Flattened size: 8 * 8 * 64 = 4096
        self.flatten_size = 8 * 8 * 64
        
        # Dense Layer: 256 hidden units
        self.fc1 = nn.Linear(self.flatten_size, hidden_dim)
        
        # Output Layer: 10 classes
        self.fc2 = nn.Linear(hidden_dim, num_classes)
        
        # Dropout rate (used during training)
        self.dropout_rate = dropout_rate
        
        # ReLU activation
        self.relu = nn.ReLU()
        
    def forward(self, x):
        """
        Forward pass through the network
        
        Args:
            x: Input tensor of shape (batch_size, in_channels, 32, 32)
            
        Returns:
            Output logits of shape (batch_size, num_classes)
        """
        # Input Layer: reshape to ensure correct dimensions
        # x should be (batch_size, 1, 32, 32) for grayscale images
        if len(x.shape) == 3:
            x = x.unsqueeze(1)  # Add channel dimension if missing
        
        # Convolutional Layer #1 + ReLU
        x = self.relu(self.conv1(x))
        
        # Pooling Layer #1
        x = self.pool1(x)  # (batch_size, 32, 16, 16)
        
        # Convolutional Layer #2 + ReLU
        x = self.relu(self.conv2(x))
        
        # Pooling Layer #2  
        x = self.pool2(x)  # (batch_size, 64, 8, 8)
        
        # Flatten for dense layers
        x = x.view(x.size(0), -1)  # (batch_size, 4096)
        
        # Dense Layer + ReLU
        x = self.relu(self.fc1(x))
        
        # Dropout (applied during training)
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        
        # Output Layer (logits)
        x = self.fc2(x)
        
        return x


def svhn_cnn_builder(model_config, local_data=None):
    """
    Build the SVHN CNN model based on the provided configuration.
    
    Args:
        model_config: Configuration object containing model parameters
        local_data: Local data (not used in this case)
        
    Returns:
        An instance of SVHNCNN
    """
    # Extract parameters from config with defaults
    in_channels = getattr(model_config, 'in_channels', 1)  # Grayscale by default
    num_classes = getattr(model_config, 'out_channels', 10)  # 10 digits
    hidden_dim = getattr(model_config, 'hidden', 256)  # Hidden layer size
    dropout_rate = getattr(model_config, 'dropout', 0.7)  # Dropout rate from notebook
    
    return SVHNCNN(in_channels=in_channels, 
                   num_classes=num_classes,
                   hidden_dim=hidden_dim,
                   dropout_rate=dropout_rate)


def call_svhn_cnn(model_config, local_data=None):
    """
    Call the model builder function to create an instance of SVHNCNN.
    
    Args:
        model_config: Configuration object containing model parameters
        local_data: Local data (not used in this case)
        
    Returns:
        An instance of SVHNCNN if model type matches, None otherwise
    """
    if model_config.type.lower() == "svhn_cnn":
        return svhn_cnn_builder(model_config, local_data)
    return None


# Register the model with the FederatedScope framework
register_model("svhn_cnn", call_svhn_cnn)


def main():
    """
    主函数：创建模型并打印所有参数名称
    """
    model = SVHNCNN(in_channels=1, num_classes=10, hidden_dim=256, dropout_rate=0.7)
    for name, param in model.named_parameters():
        print(f"    - {name}")
if __name__ == "__main__":
    main()
