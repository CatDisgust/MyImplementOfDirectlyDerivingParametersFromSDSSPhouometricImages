import torch
import numpy as np

def generate_synthetic_data(num_samples=1000, image_size=(1, 28, 28), num_classes=10, regression_range=(0, 100)):
    """
    生成模拟的SDSS图像数据、分类标签和回归目标。

    参数:
        num_samples (int): 生成样本的数量。
        image_size (tuple): 图像的尺寸，例如 (1, 28, 28) 表示单通道28x28像素。
        num_classes (int): 类别的数量。
        regression_range (tuple): 回归目标的范围。

    返回:
        X (Tensor): 生成的图像数据。
        y_classification (Tensor): 对应的分类标签。
        y_regression (Tensor): 对应的回归目标。
    """
    X = torch.randn(num_samples, *image_size)
    y_classification = torch.randint(0, num_classes, (num_samples,))
    y_regression = torch.FloatTensor(num_samples).uniform_(*regression_range)
    return X, y_classification, y_regression
