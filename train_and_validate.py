import torch
import torch.optim as optim
from convit_model import ConViT
from bayesianization import BayesianConViT
from data_generator import generate_synthetic_data
import torch.nn as nn

def train(model, optimizer, criterion, data_loader, epochs=1, device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")):
    model.train()
    for epoch in range(epochs):
        for i, (images, labels) in enumerate(data_loader):
            images, labels = images.to(device), labels.to(device)  # 将数据发送到GPU
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            if i % 10 == 0:  # 每10个批次打印一次损失
                print(f"Epoch [{epoch+1}/{epochs}], Step [{i+1}/{len(data_loader)}], Loss: {loss.item()}")

def validate(model, data_loader, device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)  # 将数据发送到GPU
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    print(f'Accuracy on the test set: {accuracy}%')

def train_with_debug(model, optimizer, criterion, data_loader, epochs=1, device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")):
    model.train()
    for epoch in range(epochs):
        for i, (images, labels) in enumerate(data_loader):
            try:
                images, labels = images.to(device), labels.to(device)  # 将数据发送到GPU
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                if i % 10 == 0:  # 每10个批次打印一次损失
                    print(f"Epoch [{epoch+1}/{epochs}], Step [{i+1}/{len(data_loader)}], Loss: {loss.item()}")
            except Exception as e:
                print(f"Exception occurred during training: {e}")
                raise

def validate_with_debug(model, data_loader, device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")):
    model.eval()
    correct = 0
    total = 0
    try:
        with torch.no_grad():
            for images, labels in data_loader:
                images, labels = images.to(device), labels.to(device)  # 将数据发送到GPU
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        accuracy = 100 * correct / total
        print(f'Accuracy on the test set: {accuracy}%')
    except Exception as e:
        print(f"Exception occurred during validation: {e}")
        raise


# 参数
num_samples = 1000
num_classes = 10
image_size = (1, 28, 28)
batch_size = 32
learning_rate = 0.001

# 设置设备
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 生成模拟数据
X, y = generate_synthetic_data(num_samples, image_size, num_classes)
X, y = X.to(device), y.to(device)  # 将数据发送到GPU

print("成功生成模拟数据")

# 创建数据加载器
dataset = torch.utils.data.TensorDataset(X, y)
data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

print("数据加载成功")

# 初始化模型、优化器和损失函数
model = BayesianConViT(num_classes=num_classes).to(device)  # 将模型发送到GPU
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()

print("模型初始化成功")

# 训练和验证模型
train_with_debug(model, optimizer, criterion, data_loader)
validate_with_debug(model, data_loader)
