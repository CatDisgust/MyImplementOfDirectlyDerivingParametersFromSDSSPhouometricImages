import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import math

# 贝叶斯线性层的实现
class BayesianLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super(BayesianLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        # 定义权重和偏置的均值和标准差
        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.Tensor(out_features))
        self.bias_sigma = nn.Parameter(torch.Tensor(out_features))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight_mu, a=math.sqrt(5))
        nn.init.constant_(self.weight_sigma, -3)
        bound = 1 / math.sqrt(self.in_features)
        nn.init.uniform_(self.bias_mu, -bound, bound)
        nn.init.constant_(self.bias_sigma, -3)

    def forward(self, x):
        weight = Normal(self.weight_mu, torch.exp(self.weight_sigma))
        bias = Normal(self.bias_mu, torch.exp(self.bias_sigma))
        return F.linear(x, weight.sample(), bias.sample())

# ConViT模型的实现
class ConViT(nn.Module):
    def __init__(self, num_classes=10, num_regression=1):
        super(ConViT, self).__init__()
        # 卷积层 - 用于提取局部特征
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)

        # Transformer层 - 用于全局依赖 (论文中提及的ConViT部分)
        self.transformer = nn.Transformer(d_model=64, nhead=8, num_encoder_layers=3, batch_first=True)

        # 贝叶斯分类头部 (论文中提及的BNN部分)
        self.classification_head = BayesianLinear(64, num_classes)

        # 回归头部
        self.regression_head = nn.Linear(64, num_regression)

    def forward(self, x):
        # 卷积层
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))

        # 调整形状以适配Transformer
        x = x.permute(0, 2, 3, 1)  # [batch, height, width, features]
        x = x.flatten(1, 2)  # [batch, seq_len, features]

        # Transformer层
        x = self.transformer(x, x)

        # 取第一个token作为输出
        x = x[:, 0, :]

        # 分类和回归头部
        classification_output = self.classification_head(x)
        regression_output = self.regression_head(x)

        return classification_output, regression_output
