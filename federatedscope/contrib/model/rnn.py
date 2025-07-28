import torch
import torch.nn as nn
from federatedscope.register import register_model

class UT_HAR_RNN(nn.Module):
    def __init__(self, hidden_dim=64):
        super(UT_HAR_RNN, self).__init__()
        self.rnn = nn.RNN(90, hidden_dim, num_layers=1)
        self.fc = nn.Linear(hidden_dim, 7)
    
    def forward(self, x):
        x = x.view(-1, 250, 90)
        x = x.permute(1, 0, 2)
        _, ht = self.rnn(x)
        outputs = self.fc(ht[-1])
        return outputs

def rnn_builder(model_config, input_shape):
    model = UT_HAR_RNN()
    return model

def call_rnn_model(model_config, local_data):
    if model_config.type == "rnn":
        return rnn_builder(model_config, local_data)

# 注册模型
register_model("rnn", call_rnn_model)

def main():
    """
    主函数：创建模型并打印所有参数名称
    """
    model = UT_HAR_RNN()
    for name, param in model.named_parameters():
        print(f"- {name}")
if __name__ == "__main__":
    main()