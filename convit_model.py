import torch
import torch.nn as nn
import torch.nn.functional as F

class ConViT(nn.Module):
    def __init__(self, num_classes=10):
        super(ConViT, self).__init__()
        # 卷积层
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)

        # Transformer层
        self.transformer = nn.Transformer(d_model=64, nhead=8, num_encoder_layers=3, batch_first=True)

        # 分类器
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        print(f"Input shape: {x.shape}")  # 打印输入形状

        # 卷积层
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))

        print(f"Post-conv shape: {x.shape}")  # 打印卷积层处理后的形状

        # 调整形状以适配Transformer
        x = x.permute(0, 2, 3, 1)  # [batch, height, width, features]
        x = x.flatten(1, 2)  # 将height和width合并，形成[batch, seq_len, features]

        # 确保特征维度符合d_model
        if x.size(2) != 64:
            raise ValueError("Feature dimension does not match d_model")

        # Transformer层
        x = self.transformer(x, x)

        print(f"Post-transformer shape: {x.shape}")  # 打印Transformer层处理后的形状

        # 取第一个token作为分类结果
        x = x[:, 0, :]

        # 分类器
        x = self.fc(x)
        return x
