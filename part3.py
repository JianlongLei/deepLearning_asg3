import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


# 1. 数据加载与预处理
def load_data(data_dir, batch_size):
    transform_32 = transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor()])
    transform_48 = transforms.Compose([transforms.Resize((48, 48)), transforms.ToTensor()])
    transform_64 = transforms.Compose([transforms.Resize((64, 64)), transforms.ToTensor()])

    dataset_32 = ImageFolder(root=data_dir, transform=transform_32)
    dataset_48 = ImageFolder(root=data_dir, transform=transform_48)
    dataset_64 = ImageFolder(root=data_dir, transform=transform_64)

    dataloader_32 = DataLoader(dataset_32, batch_size=batch_size, shuffle=True)
    dataloader_48 = DataLoader(dataset_48, batch_size=batch_size, shuffle=True)
    dataloader_64 = DataLoader(dataset_64, batch_size=batch_size, shuffle=True)

    return [dataloader_32, dataloader_48, dataloader_64]


# 2. 模型实现
class VariableResolutionCNN(nn.Module):
    def __init__(self, input_channels=3, output_channels=64, num_classes=10, pooling_type="max"):
        super(VariableResolutionCNN, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, output_channels, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        if pooling_type == "max":
            self.global_pool = nn.AdaptiveMaxPool2d((1, 1))
        elif pooling_type == "mean":
            self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        else:
            raise ValueError("Invalid pooling type. Choose 'max' or 'mean'.")

        self.fc = nn.Linear(output_channels, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


# 3. 训练与测试函数
def train_model(model, dataloaders, optimizer, criterion, device, epochs):
    model.to(device)
    history = []
    for epoch in range(epochs):
        model.train()
        total_loss, correct, total = 0, 0, 0

        for dataloader in dataloaders:
            for images, labels in dataloader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

        accuracy = 100 * correct / total
        history.append((epoch + 1, total_loss, accuracy))
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss:.4f}, Accuracy: {accuracy:.2f}%")
    return history


def test_model(model, dataloaders, criterion, device):
    model.eval()
    total_loss, correct, total = 0, 0, 0

    with torch.no_grad():
        for dataloader in dataloaders:
            for images, labels in dataloader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                total_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

    accuracy = 100 * correct / total
    print(f"Test Loss: {total_loss:.4f}, Test Accuracy: {accuracy:.2f}%")
    return total_loss, accuracy


# 4. 参数调整与结果收集
def experiment(data_dir, batch_size, epochs, output_channels, learning_rate, pooling_type):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataloaders = load_data(data_dir, batch_size)

    model = VariableResolutionCNN(output_channels=output_channels, pooling_type=pooling_type)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    print("Training...")
    train_history = train_model(model, dataloaders, optimizer, criterion, device, epochs)

    print("Testing...")
    test_loss, test_accuracy = test_model(model, dataloaders, criterion, device)

    return train_history, test_loss, test_accuracy


# 5. 数据分析与绘图
def plot_results(train_history, output_channels, test_accuracy, pooling_type):
    epochs, losses, accuracies = zip(*train_history)

    plt.figure(figsize=(14, 10))
    plt.plot(epochs, losses, label='Loss')
    plt.title(f'Training Loss (Pooling={pooling_type}, Output Channels={output_channels})')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    plt.figure(figsize=(14, 10))
    plt.plot(epochs, accuracies, label='Accuracy')
    plt.title(f'Training Accuracy (Pooling={pooling_type}, Output Channels={output_channels})')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.show()

    print(f"Final Test Accuracy: {test_accuracy:.2f}%")


# 主函数执行
if __name__ == "__main__":
    DATA_DIR = "./data/mnist-varres"  # 替换为您的数据路径
    BATCH_SIZE = 16
    EPOCHS = 10
    OUTPUT_CHANNELS = 64
    LEARNING_RATE = 0.001
    POOLING_TYPE = "mean"  # 可选择 "max" 或 "mean"

    # 执行实验
    train_hist, test_loss, test_acc = experiment(DATA_DIR, BATCH_SIZE, EPOCHS, OUTPUT_CHANNELS, LEARNING_RATE,
                                                 POOLING_TYPE)

    # 绘图分析
    plot_results(train_hist, OUTPUT_CHANNELS, test_acc, POOLING_TYPE)

    POOLING_TYPE = "max"  # 可选择 "max" 或 "mean"

    # 执行实验
    train_hist, test_loss, test_acc = experiment(DATA_DIR, BATCH_SIZE, EPOCHS, OUTPUT_CHANNELS, LEARNING_RATE,
                                                 POOLING_TYPE)

    # 绘图分析
    plot_results(train_hist, OUTPUT_CHANNELS, test_acc, POOLING_TYPE)
