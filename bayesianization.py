import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from convit_model import ConViT

class BayesianConViT(ConViT):
    def __init__(self, num_classes=10):
        super(BayesianConViT, self).__init__(num_classes)

        # 在这里，我们将模型的一部分权重转换为贝叶斯权重
        # 例如，我们可以将最后一个全连接层的权重转换为贝叶斯权重
        self.fc = BayesianLinear(64, num_classes)

class BayesianLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super(BayesianLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        # 定义权重和偏置的均值和方差
        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features))
        self.weight_rho = nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.Tensor(out_features))
        self.bias_rho = nn.Parameter(torch.Tensor(out_features))

        # 初始化参数
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight_mu, a=math.sqrt(5))
        nn.init.constant_(self.weight_rho, -3)
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight_mu)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.bias_mu, -bound, bound)
        nn.init.constant_(self.bias_rho, -3)

    def forward(self, x):
        # 计算权重和偏置的标准差
        weight_sigma = torch.log1p(torch.exp(self.weight_rho))
        bias_sigma = torch.log1p(torch.exp(self.bias_rho))

        # 从权重和偏置的分布中抽样
        weight = self.weight_mu + weight_sigma * torch.randn_like(self.weight_mu)
        bias = self.bias_mu + bias_sigma * torch.randn_like(self.bias_mu)

        return F.linear(x, weight, bias)
