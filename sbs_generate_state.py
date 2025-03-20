import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import os
import pandas as pd
import csv
from tqdm import tqdm
import yaml
from utils_LSTM import LSTMModel

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

def load_config(config_file='config.yaml'):
    with open(config_file, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config

config = load_config("config/predicate_config.yaml")

# 路径设置
feature_directory = config['paths']['save_feature_cnn_dir']
state_directory = config['paths']['clean_state_directory']
tool_directory = config['paths']['clean_tool_directory']
path_model = config['paths']['path_LSTM_model']
save_name = config['paths']['save_LSTMstate_dir']  # 预测结果保存目录

# 模型参数
window_size = config['lstm']['window_size']
lstm_parm = config['lstm']['param']

# 获取 video_id 列表（假设目录中的每个文件名或文件夹名均为视频 id）
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
        
        if tool_annotations and len(noisy_df)-1 != len(tool_annotations):
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
        
        # 目标标签：以当前帧为中心，取前后 label_window_size 帧的标签
        half_label = self.label_window_size // 2
        start_label = max(0, idx - half_label)
        end_label = min(len(self.labels), idx + half_label + 1)
        label_window = self.labels[start_label:end_label]
        one_hot = torch.zeros(len(label_window), self.num_classes)
        label_tensor = label_window.unsqueeze(1)
        one_hot.scatter_(1, label_tensor, 1)
        target = one_hot.mean(dim=0)
        
        # 如果设置了 transform，则对输入序列中每个数据点应用 transform
        if self.transform:
            input_seq = torch.stack([self.transform(frame) for frame in input_seq])
        
        return input_seq, target

def generate_label(model, video_id, device, feature_directory, state_directory, tool_directory, save_directory, batch_size=1):
    """
    利用预训练模型预测指定视频的 state，并将预测结果保存在 TXT 文件中。
    如果输入序列的长度不足 window_size，则直接写入真实状态（不调用模型推理）。
    
    参数：
      - model: 预训练好的模型（LSTMModel 实例）
      - video_id: 视频编号
      - device: 计算设备（CPU 或 GPU）
      - feature_directory: 存放 features CSV 文件的目录
      - state_directory: 存放 state 标注文件的目录
      - tool_directory: 存放 tool 标注文件的目录
      - save_directory: 保存预测结果的目录
      - batch_size: DataLoader 的 batch 大小（建议使用1）
    """
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)
    
    dataset = LSTMDataset(feature_directory, state_directory, tool_directory, video_id, window_size = window_size, param = lstm_parm, num_classes=7)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    model.eval()
    correct = 0
    total = 0
    predictions_list = []
    
    # 使用 batch_size=1 简化处理
    with torch.no_grad():
        for inputs, target in tqdm(dataloader, desc=f"Predicting labels for {video_id}"):
            # inputs shape: [1, seq_len, feature_dim]
            inputs = inputs.to(device)
            outputs = model(inputs)  # 输出 logits，形状：[1, num_classes]
            pred = torch.argmax(outputs, dim=1).item()
            predictions_list.append(pred)
            # 将 target 的 one-hot 表示转换为类别索引
            true_label = target.argmax(dim=1).item()
            
            total += 1
            if pred == true_label:
                correct += 1
    
    save_path = os.path.join(save_directory, f"{video_id}.txt")
    with open(save_path, "w", encoding="utf-8") as f:
        for pred in predictions_list:
            f.write(f"{pred}\n")
    accuracy = 100. * correct / total if total > 0 else 0.0
    print(f"视频 {video_id} 的预测结果已保存至 {save_path}")
    print(f"视频 {video_id} 的预测准确率: {accuracy:.2f}%")

    
if __name__ == '__main__':
    # 获取视频 id 列表（假设 frame 目录下的文件夹名称为视频 id）
    video_ids = os.listdir(config['paths']['frames_root'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 加载模型
    model = LSTMModel(input_size=2055, hidden_size=512, num_layers=2, num_classes=7)
    model = model.to(device)
    state_dict = torch.load(path_model, map_location=device)
    model.load_state_dict(state_dict, strict=False)
    
    # 对每个视频调用预测函数
    for video_id in video_ids:
        generate_label(model, video_id, device, feature_directory, state_directory, tool_directory, save_directory=save_name, batch_size=1)
