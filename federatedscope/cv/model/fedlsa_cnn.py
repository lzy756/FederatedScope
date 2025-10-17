"""
FedLSA论文中使用的简单CNN模型

架构:
- 2个卷积层
- 3个线性层

参考论文:
Beyond Federated Prototype Learning: Learnable Semantic Anchors with
Hyperspherical Contrast for Domain-Skewed Data (AAAI 2025)
"""

import torch.nn as nn
import torch.nn.functional as F


class FedLSACNN(nn.Module):
    """
    FedLSA的简单CNN，包含2个卷积层和3个线性层

    这与FedLSA论文中描述的架构相匹配

    网络结构:
    输入 [B, 3, 32, 32]
      ↓ Conv1(3→32) + BN + ReLU + MaxPool
    [B, 32, 16, 16]
      ↓ Conv2(32→64) + BN + ReLU + MaxPool
    [B, 64, 8, 8]
      ↓ Flatten
    [B, 4096]
      ↓ FC1(4096→hidden) + ReLU + Dropout  ← 嵌入层 z
    [B, hidden]
      ↓ FC2(hidden→hidden/2) + ReLU + Dropout
    [B, hidden/2]
      ↓ FC3(hidden/2→num_classes)
    [B, num_classes]  ← 输出logits
    """

    def __init__(self,
                 in_channels=3,
                 h=32,
                 w=32,
                 hidden=512,
                 class_num=10,
                 use_bn=True,
                 dropout=0.0):
        """
        Args:
            in_channels: 输入通道数 (3为RGB)
            h: 输入高度
            w: 输入宽度
            hidden: 隐藏维度 (嵌入维度，论文中记为I)
            class_num: 类别数 (论文中记为C)
            use_bn: 是否使用批归一化
            dropout: Dropout率
        """
        super(FedLSACNN, self).__init__()

        # 2个卷积层
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, padding=2)

        self.use_bn = use_bn
        if use_bn:
            self.bn1 = nn.BatchNorm2d(32)
            self.bn2 = nn.BatchNorm2d(64)

        # 计算卷积层后的展平大小
        # Conv1后: (h, w) → (h, w) [padding=2保持大小]
        # MaxPool1后: (h, w) → (h/2, w/2)
        # Conv2后: (h/2, w/2) → (h/2, w/2)
        # MaxPool2后: (h/2, w/2) → (h/4, w/4)
        flatten_size = (h // 4) * (w // 4) * 64

        # 3个线性层
        self.fc1 = nn.Linear(flatten_size, hidden)  # 第1个FC (编码器输出，嵌入z)
        self.fc2 = nn.Linear(hidden, hidden // 2)    # 第2个FC
        self.fc3 = nn.Linear(hidden // 2, class_num) # 最终分类器

        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(2)
        self.dropout = dropout

        # 存储维度供后续使用
        self.out_channels = hidden  # 用于投影器输入维度
        self.feature_dim = hidden    # 别名

    def forward(self, x):
        """标准前向传播"""
        # 输入验证
        if x.dim() != 4:
            raise ValueError(f"期望4D输入 [batch, channels, height, width]，"
                           f"但得到了{x.dim()}D输入，形状为{x.shape}。"
                           f"这通常意味着模型被错误的输入类型调用。")

        # 卷积层1
        x = self.conv1(x)
        if self.use_bn:
            x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # 卷积层2
        x = self.conv2(x)
        if self.use_bn:
            x = self.bn2(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # 展平
        x = x.view(x.size(0), -1)

        # FC层1 (嵌入层)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.relu(self.fc1(x))

        # FC层2
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.relu(self.fc2(x))

        # FC层3 (分类器)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.fc3(x)

        return x

    def get_embedding(self, x):
        """
        提取最终分类器之前的嵌入
        返回fc1的输出 (主嵌入层 z)

        这个方法用于FedLSA trainer中:
        1. 获取嵌入 z = get_embedding(x)
        2. 投影到超球面 h = norm(projector(z))
        3. 计算紧凑性损失 L_COM

        Returns:
            embedding: 形状 [B, hidden] 的嵌入张量
        """
        # 卷积层1
        x = self.conv1(x)
        if self.use_bn:
            x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # 卷积层2
        x = self.conv2(x)
        if self.use_bn:
            x = self.bn2(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # 展平
        x = x.view(x.size(0), -1)

        # FC层1 (这是我们的嵌入 z)
        x = F.dropout(x, p=self.dropout, training=self.training)
        embedding = self.relu(self.fc1(x))

        return embedding




