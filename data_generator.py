import torch
import numpy as np

def generate_synthetic_data(num_samples=1000, image_size=(1, 28, 28), num_classes=10):
    """
    生成模拟的SDSS图像数据和标签。

    参数:
        num_samples (int): 生成样本的数量。
        image_size (tuple): 图像的尺寸，例如 (1, 28, 28) 表示单通道28x28像素。
        num_classes (int): 类别的数量。

    返回:
        X (Tensor): 生成的图像数据。
        y (Tensor): 对应的标签。
    """
    X = torch.randn(num_samples, *image_size)
    y = torch.randint(0, num_classes, (num_samples,))
    return X, y
