import torch
import os
from tqdm import tqdm
import csv
import torch.nn as nn

def train(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    # 用 tqdm 包裹 train_loader 生成进度条
    progress_bar = tqdm(train_loader, desc="Training", leave=False)
    for batch_idx, (inputs, targets) in enumerate(progress_bar):
        # inputs shape: [batch_size, window_size, input_size]
        # targets shape: [batch_size]
        inputs, targets = inputs.to(device), targets.to(device)

        logits = model(inputs)  # logits shape: [batch_size, num_classes]
        loss = criterion(logits, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        predictions = logits.argmax(dim=1)  # 每个样本预测的类别
        total += targets.size(0)
        correct += predictions.eq(targets).sum().item()

        # 实时更新进度条描述信息
        progress_bar.set_postfix(loss=loss.item(), accuracy=100. * correct / total if total > 0 else 0)

    avg_loss = running_loss / len(train_loader)
    accuracy = 100. * correct / total if total > 0 else 0
    return avg_loss, accuracy

def test(model, test_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    criterion = nn.CrossEntropyLoss()
    # 用 tqdm 包裹 test_loader 生成进度条
    progress_bar = tqdm(test_loader, desc="Testing", leave=False)
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(progress_bar):
            inputs, targets = inputs.to(device), targets.to(device)
            logits = model(inputs)
            loss = criterion(logits, targets)

            running_loss += loss.item()
            predictions = logits.argmax(dim=1)
            total += targets.size(0)
            correct += predictions.eq(targets).sum().item()

            progress_bar.set_postfix(loss=loss.item(), accuracy=100. * correct / total if total > 0 else 0)

    avg_loss = running_loss / len(test_loader)
    accuracy = 100. * correct / total if total > 0 else 0
    return avg_loss, accuracy

# start_train 中加进度条，显示每个 epoch 中的进度
def start_train(num_epochs, save_name, model, train_loaders, test_loaders, criterion, optimizer, scheduler, device):
    os.makedirs("log", exist_ok=True)
    os.makedirs("weight", exist_ok=True)
    
    csv_file = f'log/{save_name}.csv'
    
    # 初始化 CSV 文件并写入标题
    with open(csv_file, "w", newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerow(["Epoch", "Train Loss", "Train Accuracy", "Test Loss", "Test Accuracy"])

    for epoch in range(num_epochs):
        train_losses = 0
        train_accuracies = 0
        test_losses = 0
        test_accuracies = 0
        train_samples = 0
        test_samples = 0

        # 训练部分
        # 使用 tqdm 为每个 epoch 的训练部分提供一个进度条
        for train_loader in train_loaders:
            batch_loss, batch_accuracy = train(model, train_loader, criterion, optimizer, device)
            train_losses += batch_loss * len(train_loader.dataset)  # 乘以当前批次的样本数量
            train_accuracies += batch_accuracy * len(train_loader.dataset)
            train_samples += len(train_loader.dataset)

        # 测试部分
        # 使用 tqdm 为每个 epoch 的测试部分提供一个进度条
        for test_loader in test_loaders:
            batch_loss, batch_accuracy = test(model, test_loader, criterion, device)
            test_losses += batch_loss * len(test_loader.dataset)  # 乘以当前批次的样本数量
            test_accuracies += batch_accuracy * len(test_loader.dataset)
            test_samples += len(test_loader.dataset)

        scheduler.step()

        # 计算平均损失和准确率
        train_loss = train_losses / train_samples
        train_accuracy = train_accuracies / train_samples
        test_loss = test_losses / test_samples
        test_accuracy = test_accuracies / test_samples

        # 实时写入 CSV 文件
        with open(csv_file, "a", newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=',')
            writer.writerow([epoch + 1, train_loss, train_accuracy, test_loss, test_accuracy])

        # 打印每个 epoch 的训练和测试结果
        print(f'Epoch {epoch + 1}/{num_epochs} - '
              f'Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%, '
              f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%')

        if train_accuracy > 99.999:
            break

    torch.save(model.state_dict(), f"weight/{save_name}.pth")
    print(f"训练过程已保存到 '{csv_file}' 文件中。模型权重文件已保存")
    
class LSTMModel(nn.Module):
    def __init__(self, input_size=2055, hidden_size=512, num_layers=2, num_classes=7):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # x 的形状：[batch_size, seq_len, input_size]
        lstm_out, _ = self.lstm(x)
        # 选择最后一个时间步的输出
        if lstm_out.dim() == 2:
            last_output = lstm_out
        else:
            last_output = lstm_out[:, -1, :]
        out = self.fc(last_output)
        return out