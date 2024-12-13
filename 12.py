import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt
import os
from torchvision import datasets, transforms
from torch import optim, nn
from torch.utils.data import DataLoader, random_split
from sklearn.model_selection import train_test_split

def set_random_seed(seed=77):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)

def get_data_loaders(data_dir, batch_size=32):
    """Prepare data loaders for training, validation, and testing."""
    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    train_data = datasets.ImageFolder(os.path.join(data_dir, 'train'), transform=transform)
    test_data = datasets.ImageFolder(os.path.join(data_dir, 'test'), transform=transform)

    train_size = len(train_data) - 10000
    val_size = 10000
    traindata, valdata = random_split(train_data, [train_size, val_size], generator=torch.Generator().manual_seed(42))

    trainloader = DataLoader(traindata, batch_size=batch_size, shuffle=True, num_workers=2)
    valloader = DataLoader(valdata, batch_size=batch_size, shuffle=False, num_workers=2)
    testloader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=2)

    return trainloader, valloader, testloader

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.fc = nn.Linear(64 * 3 * 3, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

def train_one_epoch(dataloader, model, loss_func, optimizer, device):
    """Train the model for one epoch."""
    model.train()
    total_loss, correct, total = 0, 0, 0
    for X, y in dataloader:
        X, y = X.to(device), y.to(device)

        pred = model(X)
        loss = loss_func(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        correct += (pred.argmax(1) == y).type(torch.float).sum().item()
        total += y.size(0)

    return total_loss / len(dataloader), 100 * correct / total

def evaluate(dataloader, model, loss_func, device):
    """Evaluate the model on validation or test data."""
    model.eval()
    total_loss, correct, total = 0, 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            loss = loss_func(pred, y)

            total_loss += loss.item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
            total += y.size(0)

    return total_loss / len(dataloader), 100 * correct / total

def plot_metrics(epochs, train_metrics, val_metrics, metric_name):
    """Plot metrics over epochs."""
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, epochs + 1), train_metrics, label=f'Train {metric_name}')
    plt.plot(range(1, epochs + 1), val_metrics, label=f'Validation {metric_name}')
    plt.xlabel('Epoch')
    plt.ylabel(metric_name)
    plt.legend()
    plt.title(f'{metric_name} over Epochs')
    plt.show()

def main():
    set_random_seed()
    data_dir = 'data/mnist-varres'
    batch_size = 32
    epochs = 10

    trainloader, valloader, testloader = get_data_loaders(data_dir, batch_size)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = CNN().to(device)
    loss_func = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train_losses, train_accuracies = [], []
    val_losses, val_accuracies = [], []

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        train_loss, train_acc = train_one_epoch(trainloader, model, loss_func, optimizer, device)
        val_loss, val_acc = evaluate(valloader, model, loss_func, device)

        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)

        print(f"Train Loss: {train_loss:.4f}, Accuracy: {train_acc:.2f}% | "
              f"Validation Loss: {val_loss:.4f}, Accuracy: {val_acc:.2f}%")

    print("\nFinal Evaluation on Test Set")
    test_loss, test_acc = evaluate(testloader, model, loss_func, device)
    print(f"Test Loss: {test_loss:.4f}, Accuracy: {test_acc:.2f}%")

    plot_metrics(epochs, train_losses, val_losses, 'Loss')
    plot_metrics(epochs, train_accuracies, val_accuracies, 'Accuracy')

if __name__ == '__main__':
    main()
