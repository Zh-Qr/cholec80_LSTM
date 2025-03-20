import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset
import random
import os
import pandas as pd
from PIL import Image
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torchvision")


def split_files_and_extract_frames(split_dir, frame_store, train_ratio=0.8):
    """
    根据是否存在划分文件来决定如何划分数据集，并根据划分结果从frame_store中提取训练集和测试集的帧数据。

    参数:
    - split_dir: 划分文件保存的目录。
    - frame_store: VideoFrameStore实例，包含数据。
    - train_ratio: 训练集占比，默认70%。
    
    返回：
    - train_frames: 训练集的帧数据
    - test_frames: 测试集的帧数据
    """
    # 定义划分文件路径
    train_file = os.path.join(split_dir, "train_videos.txt")
    test_file = os.path.join(split_dir, "test_videos.txt")

    if os.path.exists(train_file) and os.path.exists(test_file):
        # 加载训练集视频
        with open(train_file, "r") as f:
            train_videos = [line.strip() for line in f.readlines()]
        # 加载测试集视频
        with open(test_file, "r") as f:
            test_videos = [line.strip() for line in f.readlines()]
        # print("加载已有划分文件完成。")
    else:
        # 获取所有视频ID
        video_ids = list(frame_store.data.keys())  # 获取frame_store中所有的视频ID
        
        # 打乱视频ID列表
        random.shuffle(video_ids)
        
        # 按照train_ratio将数据划分为训练集和测试集
        train_size = int(len(video_ids) * train_ratio)
        train_videos = video_ids[:train_size]
        test_videos = video_ids[train_size:]
        
        # 保存训练集和测试集划分结果
        if not os.path.exists(split_dir):
            os.makedirs(split_dir)
        
        # 保存训练集
        with open(train_file, "w") as f:
            for video in train_videos:
                f.write(f"{video}\n")
        
        # 保存测试集
        with open(test_file, "w") as f:
            for video in test_videos:
                f.write(f"{video}\n")

        print(f"划分文件已保存，训练集大小: {len(train_videos)}, 测试集大小: {len(test_videos)}")

    # 根据划分结果从frame_store提取训练集和测试集的帧数据
    train_frames = {}
    test_frames = {}

    # 提取训练集帧数据
    for video_id in train_videos:
        if video_id in frame_store.data:
            train_frames[video_id] = frame_store.data[video_id]

    # 提取测试集帧数据
    for video_id in test_videos:
        if video_id in frame_store.data:
            test_frames[video_id] = frame_store.data[video_id]

    return train_frames, test_frames

"""
加载数据class类
"""
class VideoFrameStore:
    def __init__(self, directory, use_noise=False, noise_directory=None):
        """
        directory: 根目录，包含frames和annotations子目录
        use_noise: 是否使用噪声数据
        noise_directory: 噪声数据的目录路径，如果use_noise为True则需要指定
        """
        self.directory = directory
        self.frames_directory = os.path.join(directory, "frames")
        if use_noise and noise_directory:
            self.annotations_directory = noise_directory
        else:
            self.annotations_directory = os.path.join(directory, "annotations")
        self.data = self.load_data()
        self.phase_to_int = {
            "Preparation": 0,
            "CalotTriangleDissection": 1,
            "ClippingCutting": 2,
            "GallbladderDissection": 3,
            "GallbladderPackaging": 4,
            "CleaningCoagulation": 5,
            "GallbladderRetraction": 6
        }
        # 创建数字到阶段名称的映射
        self.int_to_phase = {v: k for k, v in self.phase_to_int.items()}

    def load_data(self):
        """
        加载所有视频帧及其对应的状态信息。
        返回字典格式: {video_id: {frame_number: (frame_path, state)}}
        """
        data = {}
        video_ids = [d for d in os.listdir(self.frames_directory) if os.path.isdir(os.path.join(self.frames_directory, d))]
        for video_id in video_ids:
            video_path = os.path.join(self.frames_directory, video_id)
            annotation_path = os.path.join(self.annotations_directory, f"{video_id}.txt")
            numbers = []
            with open(annotation_path, 'r') as file:
                for line in file:
                    number = int(line.strip())
                    numbers.append(number)
                    
            frame_files = sorted(os.listdir(video_path))
            frame_paths = [os.path.join(video_path, f) for f in frame_files]
            
            data[video_id] = {}
            for idx, frame_path in enumerate(frame_paths):
                state = numbers[idx] if idx < len(numbers) else None
                data[video_id][idx] = (frame_path, state)
        return data

    def get_frame_info(self, video_id, frame_number):
        """
        通过视频ID和帧号获取帧的路径和状态。
        """
        video_data = self.data.get(video_id, {})
        return video_data.get(frame_number, (None, None))

    def show_frame(self, video_id, frame_number):
        """
        展示指定视频的指定帧。
        """
        frame_path, state = self.get_frame_info(video_id, frame_number)
        if frame_path:
            image = Image.open(frame_path)
            phase_name = self.int_to_phase.get(state, "Unknown Phase")
            plt.imshow(image)
            plt.title(f"Phase: {phase_name}")
            plt.axis('off')  # Hide axes
            plt.show()
        else:
            print("Frame not found.")
            
"""
通过继承Dataset生成数据集
"""
class VideoFrameDataset(Dataset):
    def __init__(self, video_frame_store, transform=None):
        """
        video_frame_store: 已加载的视频帧数据，应该是一个字典形式的数据结构，
                           例如：{'video_id': {'frame_id': (frame_path, state), ...}, ...}
        transform: 对图像进行转换的操作，默认为None。
        """
        self.video_frame_store = video_frame_store  # 使用VideoFrameStore实例
        self.transform = transform
        self.data = self.load_data()  # 加载数据

    def load_data(self):
        """
        遍历video_frame_store，加载每个视频帧的路径和对应的状态信息。
        返回列表格式: [(frame_path, state)]
        """
        data = []
        
        for video_id, frames in self.video_frame_store.items():
            for frame_id, (frame_path, state) in frames.items():
                # 将每个帧和状态信息添加到数据集中
                data.append((frame_path, state))
        
        return data

    def __len__(self):
        """
        返回数据集的大小
        """
        return len(self.data)

    def __getitem__(self, idx):
        """
        获取指定索引的数据，包括图像和标签。
        每次调用时按需加载图像。
        """
        frame_path, state = self.data[idx]
        
        # 加载图像
        frame_data = Image.open(frame_path).convert('RGB')  # 直接加载并转换为RGB图像
        
        # 如果指定了transform，应用转换
        if self.transform:
            frame_data = self.transform(frame_data)
        
        # 转换状态为tensor
        state = torch.tensor(state, dtype=torch.long)
        
        return frame_data, state
    
"""
获取LSTM训练类，包括滑动窗口大小设置
"""
class LSTMDataset(Dataset):
    def __init__(self, directory, video_id, transform=None, window_size=100):
        """
        directory: 数据所在目录，
        video_id: 视频编号
        transform: 数据变换
        window_size: 滑动窗口大小，即连续输入的帧数，模型将以这些帧预测下一帧状态
        """
        self.video_id = video_id
        self.video_csv_path = os.path.join(directory, 'extracted_features_csv')
        self.annotations_directory = os.path.join(directory, 'annotations')
        self.transform = transform
        self.window_size = window_size

        self.features = self.load_feature()  # 加载整个视频的特征，形状为 [num_frames, feature_dim]
        self.labels = self.load_label()        # 加载整个视频的标签

        # 将 features 从 list 转换为 tensor，便于后续索引，假设每帧特征的维度保持一致
        self.features = torch.stack(self.features, dim=0)  # shape: [num_frames, feature_dim]
        self.labels = torch.tensor(self.labels, dtype=torch.long)

    def load_feature(self):
        feature = []
        feature_path = os.path.join(self.video_csv_path, f'{self.video_id}_features.csv')
        feature_file = pd.read_csv(feature_path, header=0)
        for _, row in feature_file.iterrows():
            # 忽略 row 中最后两列的数据
            feature.append(torch.tensor(row.values[:-2], dtype=torch.float32))
        return feature

    def load_label(self):
        label = []
        label_path = os.path.join(self.annotations_directory, f'{self.video_id}.txt')
        with open(label_path, 'r') as file:
            lines = file.readlines()
            for line in lines:
                label.append(int(line.strip()))
        return label

    def __len__(self):
        # 每个样本由连续 window_size 帧作为输入，目标为 window_size 后的一帧状态
        # 总样本数 = 总帧数 - window_size
        return len(self.features) - self.window_size

    def __getitem__(self, idx):
        # 输入：从 idx 开始的连续 window_size 帧
        input_seq = self.features[idx: idx + self.window_size]  # shape: [window_size, feature_dim]
        # 目标：紧跟输入窗口的下一帧状态
        target = self.labels[idx + self.window_size]

        if self.transform:
            input_seq = self.transform(input_seq)

        return input_seq, target
    
    
# 获取预测的时间序列信息和标签对比的准确度
def get_accuracy(labels, states):
    length = len(labels)
    accuracy = 0
    for i in range(length):
        if labels[i] == states[i]:
         lstm_modelccuracy += 1
    ration = accuracy/length
    print(f"准确率：{ration:.4f}")
    return ration

# Mixup 数据增强方法
def mixup_data(x, y, device, alpha=1.0):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam