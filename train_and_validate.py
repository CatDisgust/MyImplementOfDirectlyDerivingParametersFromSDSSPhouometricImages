import torch
import torch.optim as optim
from convit_model import ConViT
import torch.nn as nn
from data_generator import generate_synthetic_data


# 训练函数
def train(model, optimizer, criterion_classification, criterion_regression, data_loader, epochs=10,
          device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")):
    model.train()
    for epoch in range(epochs):
        for i, (images, labels, regression_targets) in enumerate(data_loader):
            images, labels, regression_targets = images.to(device), labels.to(device), regression_targets.to(device)

            optimizer.zero_grad()
            classification_outputs, regression_outputs = model(images)

            # 调整回归输出的尺寸
            regression_outputs = torch.squeeze(regression_outputs)

            # 计算分类和回归损失
            loss_classification = criterion_classification(classification_outputs, labels)
            loss_regression = criterion_regression(regression_outputs, regression_targets)

            # 综合两种损失
            loss = loss_classification + loss_regression

            loss.backward()
            optimizer.step()

            if i % 10 == 0:
                print(f"Epoch [{epoch + 1}/{epochs}], Step [{i + 1}/{len(data_loader)}], Loss: {loss.item()}")


# 验证函数
def validate(model, data_loader, device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")):
    model.eval()
    correct_classification = 0
    total_classification = 0
    total_regression_error = 0

    with torch.no_grad():
        for images, labels, regression_targets in data_loader:
            images, labels, regression_targets = images.to(device), labels.to(device), regression_targets.to(device)
            classification_outputs, regression_outputs = model(images)

            # 分类准确率
            _, predicted = torch.max(classification_outputs.data, 1)
            total_classification += labels.size(0)
            correct_classification += (predicted == labels).sum().item()

            # 回归误差
            regression_error = torch.mean((regression_outputs - regression_targets) ** 2)
            total_regression_error += regression_error.item()

    print(f'Classification Accuracy: {100 * correct_classification / total_classification}%')
    print(f'Average Regression Error: {total_regression_error / len(data_loader)}')


# 参数配置
num_samples = 1000
num_classes = 10
image_size = (1, 28, 28)
batch_size = 16
learning_rate = 0.001
num_epochs = 1

# 生成模拟数据
X, y_classification, y_regression = generate_synthetic_data(num_samples, image_size, num_classes)

# 创建数据加载器
dataset = torch.utils.data.TensorDataset(X, y_classification, y_regression)
data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

# 初始化模型
model = ConViT(num_classes=num_classes).to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))

# 优化器和损失函数
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion_classification = nn.CrossEntropyLoss()
criterion_regression = nn.MSELoss()

# 训练和验证模型
train(model, optimizer, criterion_classification, criterion_regression, data_loader, epochs=num_epochs)
validate(model, data_loader)
