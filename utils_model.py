import torch
from torchvision import models
import os
from PIL import Image
import torch.nn as nn
from tqdm import tqdm
import csv
import warnings
import utils_data
warnings.filterwarnings("ignore", category=UserWarning, module="torchvision")

#特征提取模型
class ResNet50V2FeatureExtractor(nn.Module):
    def __init__(self, num_classes, pretrained=True):
        super(ResNet50V2FeatureExtractor, self).__init__()
        
        # 加载预训练的 ResNet50 V2 模型
        resnet50v2 = models.resnet50(pretrained=pretrained)
        
        # 去除 ResNet 的最后全连接层（即原来用于分类的部分）
        self.features = nn.Sequential(*list(resnet50v2.children())[:-1])
        
        # 自定义一个新的分类头
        self.fc = nn.Linear(resnet50v2.fc.in_features, num_classes)
    
    def forward(self, x):
        # 提取特征，去掉最后的分类层
        x = self.features(x)
        
        # 平均池化处理：自适应池化成1x1的大小，再将维度展平
        feature = torch.flatten(x, 1)
        
        # 分类预测
        outputs = self.fc(feature)
        
        return feature, outputs  # 返回2048维特征和分类预测
    
#时间序列模型
class LSTMModel(nn.Module):
    def __init__(self, input_size=2050, hidden_size=512, num_layers=2, num_classes=7):
        super(LSTMModel, self).__init__()

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # x 的形状：[batch_size, seq_len, input_size]
        lstm_out, (h_n, c_n) = self.lstm(x)

        # 如果是二维张量，直接使用它作为 LSTM 输出
        if len(lstm_out.shape) == 2:
            last_lstm_output = lstm_out
        else:
            # 选择最后一个时间步的输出
            last_lstm_output = lstm_out[:, -1, :]

        # 全连接层进行分类
        out = self.fc(last_lstm_output)
        
        return out
    
#加载模型
def load_model(device, CNN_path = None, LSTM_path = None, pretrained = False):
    CNN_model = ResNet50V2FeatureExtractor(num_classes=7, pretrained=True)
    CNN_model = CNN_model.to(device)
    lstm_model = LSTMModel()
    lstm_model = lstm_model.to(device)
    
    if pretrained:
        try:
            state_dict = torch.load(CNN_path)
            CNN_model.load_state_dict(state_dict, strict=False)
        except:
            print("CNN Load Error")
        try:
            state_dict = torch.load(LSTM_path)
            lstm_model.load_state_dict(state_dict, strict=False)
        except:
            print("LSTM Load Error")
    return CNN_model, lstm_model

# 带进度条的 test 函数
def test(model, test_loader, criterion,device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    # 使用 tqdm 包装 test_loader
    loop = tqdm(enumerate(test_loader), total=len(test_loader), desc="Testing")
    
    with torch.no_grad():
        for batch_idx, (inputs, targets) in loop:
            inputs, targets = inputs.to(device), targets.to(device)

            # 前向传播
            _, outputs = model(inputs)
            loss = criterion(outputs, targets)

            # 累加损失和正确预测数量
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            # 更新进度条后缀显示
            loop.set_postfix({
                "loss": running_loss / (batch_idx + 1),  # 平均损失
                "accuracy": 100. * correct / total      # 当前准确率
            })

    # 计算整个测试集的平均损失和准确率
    avg_loss = running_loss / len(test_loader)
    accuracy = 100. * correct / total
    return avg_loss, accuracy

def train(model, train_loader, criterion, optimizer, epoch, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    # 使用 tqdm 包装 train_loader
    loop = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch+1} Training")

    for batch_idx, (inputs, targets) in loop:
        inputs, targets = inputs.to(device), targets.to(device)

        _, outputs = model(inputs)
        loss = criterion(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        # 更新进度条后缀显示
        loop.set_postfix({
            "loss": running_loss / (batch_idx + 1),  # 平均损失
            "accuracy": 100. * correct / total      # 当前准确率
        })                                

    avg_loss = running_loss / len(train_loader)
    accuracy = 100. * correct / total
    return avg_loss, accuracy

def train_ITLM(model, train_loader, criterion, optimizer, epoch, prune_ratio, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    all_losses = []
    prune_indices = set()

    loop = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch+1} Training")
    for batch_idx, (inputs, targets) in loop:
        if batch_idx in prune_indices:
            continue  # 跳过裁剪的样本

        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        all_losses.append((batch_idx, loss.item()))

        loop.set_postfix(loss=total_loss / (batch_idx + 1), accuracy=100. * correct / total)

        # 在每个epoch结束时处理裁剪
        num_prune = int(prune_ratio * len(all_losses))
        sorted_losses = sorted(all_losses, key=lambda x: x[1], reverse=True)
        prune_indices = {idx for idx, _ in sorted_losses[:num_prune]}

    avg_loss = total_loss / (len(train_loader) - len(prune_indices))
    accuracy = 100. * correct / total
    return avg_loss, accuracy

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

def train_mixup(model, train_loader, criterion, optimizer, epoch, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    # 使用 tqdm 包装 train_loader
    loop = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch+1} Training")

    for batch_idx, (inputs, targets) in loop:
        inputs, targets = inputs.to(device), targets.to(device)
        
        inputs, targets_a, targets_b, lam = utils_data.mixup_data(inputs, targets, device)
        inputs, targets_a, targets_b = map(torch.autograd.Variable, (inputs, targets_a, targets_b))

        _, outputs = model(inputs)
        loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        # 更新进度条后缀显示
        loop.set_postfix({
            "loss": running_loss / (batch_idx + 1),  # 平均损失
            "accuracy": 100. * correct / total      # 当前准确率
        })                                

    avg_loss = running_loss / len(train_loader)
    accuracy = 100. * correct / total
    return avg_loss, accuracy

def start_train(num_epochs, save_name, model, train_loader, test_loader, criterion, optimizer, scheduler, device, train_tactic = None):
    os.makedirs("log",exist_ok=True)
    os.makedirs("weight",exist_ok=True)
    
    csv_file = f'log/{save_name}.csv'
    
    # 初始化 CSV 文件并写入标题
    with open(csv_file, "w", newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerow(["Epoch", "Train Loss", "Train Accuracy", "Test Loss", "Test Accuracy"])

    
    for epoch in range(num_epochs):
        if train_tactic == "mixup":
            train_loss, train_accuracy = train_mixup(model, train_loader, criterion, optimizer, epoch, device)
        elif train_tactic == "ITLM":
            train_loss, train_accuracy = train_ITLM(model, train_loader, criterion, optimizer, epoch, device)
        else:
            train_loss, train_accuracy = train(model, train_loader, criterion, optimizer, epoch, device)
            
        test_loss, test_accuracy = test(model, test_loader, criterion, device)
        scheduler.step()
        
         # 实时写入 CSV 文件
        with open(csv_file, "a", newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=',')
            writer.writerow([epoch + 1, train_loss, train_accuracy, test_loss, test_accuracy])
            # writer.writerow(epoch +1, train_loss, train_accuracy, test_loss, test_accuracy)


        # 打印每个 epoch 的训练和测试结果
        print(f'Epoch {epoch + 1}/{num_epochs} - '
            f'Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%, '
            f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%')
    
        if train_accuracy > 99.9:
            break
    
    torch.save(model.state_dict(), f"weight/{save_name}.pth")
    print(f"训练过程已保存到 '{csv_file}' 文件中。模型权重文件已保存")
    
# 获取提取的特征
def extract_features_with_accuracy(frame_store, save_dir, model, transform, device):
    """
    从给定的视频帧数据中提取特征并保存到CSV文件，同时计算每个视频的预测准确率。
    
    参数:
    - frame_store: VideoFrameStore 实例，包含视频帧及其状态。
    - save_dir: 保存特征数据的目录。
    - model2048: 用于提取特征的预训练模型（ResNet50V2）。
    - model: 用于分类的模型（可能是一个7类分类模型）。
    - transform: 用于图像预处理的转换操作。
    - device: 计算设备（CPU或GPU）。
    """
    model.eval()  # 将模型设置为评估模式
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # 遍历视频数据，逐个处理视频
    for video_id, frames in frame_store.data.items():
        video_feature_file = os.path.join(save_dir, f"{video_id}_features.csv")
        correct_predictions = 0
        total_frames = 0
        
        # 打开CSV文件并写入标题
        with open(video_feature_file, mode='w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([f"Feature_{i}" for i in range(2048)] + ["Predicted_Label", "Actual_Label"])  # 添加标签列

            # 使用tqdm包装数据加载的过程，显示进度条
            frame_items = list(frames.items())
            loop = tqdm(frame_items, desc=f"Extracting features for {video_id}", unit="frame")
            
            for frame_number, (frame_path, state) in loop:
                if frame_path:  # 确保帧路径存在
                    # 加载并转换图像
                    image = Image.open(frame_path).convert("RGB")
                    image = transform(image).unsqueeze(0).to(device)

                    # 提取特征和预测标签
                    with torch.no_grad():
                        feature, outputs = model(image)  # 使用model进行一次推理，返回特征和预测结果
                    
                    feature = feature.view(-1).cpu().numpy()  # 展平特征为一维

                    # 获取预测标签
                    _, predicted_label = torch.max(outputs, 1)

                    # 更新准确率
                    if predicted_label.item() == state:  # 如果预测标签等于实际标签
                        correct_predictions += 1
                    total_frames += 1
                    
                    # 将特征和标签写入CSV
                    writer.writerow(list(feature) + [predicted_label.item(), state])  # 使用item()获取标量值
                
                # 更新进度条
                loop.set_postfix({"Processed": len(loop)})  # 更新进度条显示
        
        # 计算视频准确率
        if total_frames > 0:
            accuracy = correct_predictions / total_frames
        else:
            accuracy = 0

        print(f"视频 {video_id} 的特征已保存到 {video_feature_file}. 准确率: {accuracy:.4f}")