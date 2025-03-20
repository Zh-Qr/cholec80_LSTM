import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import Dataset
import os
import datetime
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import pandas as pd
import csv
from tqdm import tqdm
import yaml
from utils_LSTM import LSTMModel

def load_config(config_file='config.yaml'):
    with open(config_file, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config

config = load_config("config/robust_config.yaml")

# 路径设置
feature_directory = config['paths']['save_feature_cnn_dir']
state_directory = config['paths']['save_LSTMstate_dir']
clean_state_directory = config['paths']['clean_state_directory']
tool_directory = config['paths']['save_tool_dir']
save_name = config['paths']['save_LSTMGCE_weight']

# 模型参数
window_size = config['lstm']['window_size']
lstm_parm = config['lstm']['param']

# 获取 video_id 列表
def get_videoid(directory):
    video_ids = []
    with open(directory, 'r') as file:
        for line in file:
            video_id = line.strip()
            video_ids.append(video_id)
    return video_ids

    
class LSTMDataset(Dataset):
    def __init__(self, feature_directory, state_directory, tool_directory, video_id, 
                 transform=None, window_size=100, param = 5, num_classes=7):
        """
        Args:
          - feature_directory: 存放 features CSV 文件的目录（用于加载 noisy 特征）
          - state_directory: 存放 state 标注文件的目录（真实标签）
          - tool_directory: 存放 tool 标注文件的目录
          - video_id: 视频编号
          - transform: 数据变换（例如对原始数据进行处理，如果需要的话）
          - window_size: 用于计算输入序列的窗口大小（取当前帧之前最多 window_size 帧）
          - num_classes: 状态类别数
        """
        self.video_id = video_id
        self.video_csv_path = feature_directory
        self.annotations_directory = state_directory
        self.tool_directory = tool_directory
        self.transform = transform
        self.window_size = window_size
        self.num_classes = num_classes
        # 定义 label 窗口大小为 window_size 的 1/5
        self.label_window_size = window_size // param
    

        self.features = self.load_feature()  # 加载并合并特征
        self.labels = self.load_label()        # 加载所有标签
        
        # 保证 features 与 labels 数量一致（取两者的最小数量）
        num_samples = min(len(self.features), len(self.labels))
        self.features = self.features[:num_samples]
        self.labels = self.labels[:num_samples]
        
        # 转换为 tensor
        self.features = torch.stack(self.features, dim=0)  # [num_frames, feature_dim]
        self.labels = torch.tensor(self.labels, dtype=torch.long)

    def load_feature(self):
        features = []
        feature_path = os.path.join(self.video_csv_path, f'{self.video_id}_features.csv')
        noisy_df = pd.read_csv(feature_path, header=0)
        
        tool_path = os.path.join(self.tool_directory, f'{self.video_id}.txt')
        tool_annotations = []
        if os.path.exists(tool_path):
            with open(tool_path, 'r', encoding='utf-8') as f:
                reader = csv.reader(f, delimiter='\t')
                next(reader, None)
                for row in reader:
                    if len(row) < 7:
                        continue
                    try:
                        tool_values = [int(x) for x in row[:7]]
                    except Exception as e:
                        print(f"解析 tool 文件 {tool_path} 时出错：{e}，行内容：{row}")
                        continue
                    tool_annotations.append(torch.tensor(tool_values, dtype=torch.float))
        else:
            print(f"视频 {self.video_id} 的 tool 文件 {tool_path} 不存在，仅使用 noisy 特征。")
        
        if tool_annotations and len(noisy_df) != len(tool_annotations):
            print(f"视频 {self.video_id} 中 noisy 特征数量 ({len(noisy_df)}) 与 tool 标签行数 ({len(tool_annotations)}) 不一致，按最小数量匹配。")
        sample_num = len(noisy_df) if not tool_annotations else min(len(noisy_df), len(tool_annotations))
        
        for i in range(sample_num):
            noisy_feature = torch.tensor(noisy_df.iloc[i], dtype=torch.float32)
            if tool_annotations:
                tool_feature = tool_annotations[i]  # 7 维
                combined_feature = torch.cat([noisy_feature, tool_feature], dim=0)  # 2055 维
            else:
                combined_feature = noisy_feature
            features.append(combined_feature)
        return features

    def load_label(self):
        labels = []
        label_path = os.path.join(self.annotations_directory, f'{self.video_id}.txt')
        with open(label_path, 'r') as file:
            for line in file:
                labels.append(int(line.strip()))
        return labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        # 输入序列：取当前帧之前最多 window_size 帧
        start_input = max(0, idx - self.window_size)
        input_seq = self.features[start_input:idx]  # shape: [num_input_frames, feature_dim]
        # 如果没有历史帧，则直接使用当前帧
        if input_seq.shape[0] == 0:
            input_seq = self.features[idx].unsqueeze(0)
        # 如果不足 window_size，则复制扩充到正好 window_size
        if input_seq.shape[0] < self.window_size:
            repeat_factor = (self.window_size + input_seq.shape[0] - 1) // input_seq.shape[0]
            input_seq = input_seq.repeat((repeat_factor, 1))[:self.window_size]
            # 再对扩充后的序列求平均，得到一个平均特征
            avg_feature = input_seq.mean(dim=0, keepdim=True)
            # 将平均特征重复 window_size 次，确保每一帧都相同
            input_seq = avg_feature.repeat(self.window_size, 1)
        
      
        target = self.labels[idx]
        
        # 如果设置了 transform，则对输入序列中每个数据点应用 transform
        if self.transform:
            input_seq = torch.stack([self.transform(frame) for frame in input_seq])
        
        return input_seq, target

def train(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    progress_bar = tqdm(train_loader, desc="Training", leave=False)
    for _, (inputs, targets) in enumerate(progress_bar):
        # inputs: [batch_size, seq_len, feature_dim]
        # targets: [batch_size, num_classes]（平均的 one-hot 向量）
        inputs, targets = inputs.to(device), targets.to(device)
        logits = model(inputs)  # logits: [batch_size, num_classes]
        loss = criterion(logits, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        # 计算预测正确率：取 logits 的 argmax 和 targets 的 argmax
        predictions = logits.argmax(dim=1)
        total += targets.size(0)
        correct += predictions.eq(targets).sum().item()
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
    
class GCELoss(nn.Module):
    def __init__(self, q=0.7, eps=1e-6):
        super(GCELoss, self).__init__()
        assert 0 <= q <= 1, "q必须在 [0,1] 范围内"
        self.q = q
        self.eps = eps

    def forward(self, logits, targets):
        # logits: [batch_size, num_classes]
        # targets: [batch_size]，目标类别标签
        probs = torch.softmax(logits, dim=1)
        p = probs.gather(1, targets.unsqueeze(1)).squeeze(1).clamp(min=self.eps)
        if self.q == 0:
            loss = -torch.log(p)
        else:
            loss = (1 - p ** self.q) / self.q
        return loss.mean()

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 加载模型
    lstm_model = LSTMModel(input_size=2055, hidden_size=512, num_layers=2, num_classes=7)
    lstm_model = lstm_model.to(device)
    
    # 构造训练数据集（这里只示范训练部分）
    train_videoids = get_videoid(config['robust_loss']['train_splite_dir'])
    test_videoids = get_videoid(config['robust_loss']['test_splite_dir'])
    train_datasets = []
    for vid in train_videoids:
        dataset = LSTMDataset(feature_directory, state_directory, tool_directory, vid, window_size=window_size, param=lstm_parm, num_classes=7)
        train_datasets.append(dataset)
    print("训练数据集加载完成")
    
    test_datasets = []
    for vid in test_videoids:
        dataset = LSTMDataset(feature_directory, clean_state_directory, tool_directory, vid, window_size=window_size, param=lstm_parm, num_classes=7)
        test_datasets.append(dataset)
    print("测试数据集加载完成")
    
    train_loaders = [DataLoader(ds, batch_size=config['lstm']['batch_size'], shuffle=True, num_workers=4) for ds in train_datasets]
    test_loaders = [DataLoader(ds, batch_size=config['lstm']['batch_size'], shuffle=True, num_workers=4) for ds in test_datasets]
    
    # 定义损失函数与优化器，这里使用 SmoothBootStrap 作为损失函数
    criterion = GCELoss()
    optimizer = optim.Adam(lstm_model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    
    try:
        start_train(num_epochs=config['lstm']['num_epochs'], save_name=save_name, model=lstm_model,
                    train_loaders=train_loaders, test_loaders=test_loaders, criterion=criterion,
                    optimizer=optimizer, scheduler=scheduler, device=device)
    except KeyboardInterrupt:
        print("训练被中断，正在保存模型权重...")
        torch.save(lstm_model.state_dict(), f"weight/{save_name}.pth")
        print(f"模型权重已保存为 weight/{save_name}.pth")