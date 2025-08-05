import torch
import torch.nn as nn
import torch.nn.functional as F
from federatedscope.register import register_model

class FedSAK_MNIST_Model(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=100, num_classes=10):
        super(FedSAK_MNIST_Model, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = x.view(-1, 784)  # 展平：28x28 → 784
        x = F.relu(self.fc1(x))
        x = self.fc2(x)  # 不使用 softmax（交叉熵内部已包含）
        return x
    
def fedsak_mnist_builder(model_config, input_shape):
    model = FedSAK_MNIST_Model()
    return model

def call_fedsak_mnist_model(model_config, local_data):
    if model_config.type == "fedsak_mnist":
        return fedsak_mnist_builder(model_config, local_data)

# 注册模型
register_model("fedsak_mnist", call_fedsak_mnist_model)

def main():
    """
    主函数：创建模型并打印所有参数名称
    """
    model = FedSAK_MNIST_Model()
    for name, param in model.named_parameters():
        print(f"- {name}")
if __name__ == "__main__":
    main()