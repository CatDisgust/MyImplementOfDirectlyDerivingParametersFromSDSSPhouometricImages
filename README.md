# 直接从SDSS光度图像推导恒星参数

## 项目概述
此项目旨在实现论文《直接从SDSS光度图像推导恒星参数》中描述的方法，通过使用深度学习技术直接从SDSS（Sloan Digital Sky Survey）光度图像中提取恒星大气参数。项目采用了一个结合了卷积神经网络和Transformer的贝叶斯ConViT模型，旨在为所有预测提供相应的置信水平。

## 主要特性
- **ConViT模型**：一种结合了卷积层和Transformer层的深度学习模型。
- **贝叶斯近似**：模型的一部分使用了近似贝叶斯方法，以处理不确定性。
- **分类和回归任务**：模型包含用于分类和回归任务的独立头部。

## 安装指南
运行以下代码下载依赖项目：

```shell
pip install -r requirements.txt
```

## 使用指南
运行以下命令来运行项目
```shell
python train_and_validate.py
```


贡献者
[xuaofeng]

许可证
[选择合适的许可证，例如MIT]

