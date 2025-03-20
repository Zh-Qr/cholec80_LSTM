import torch
from torchvision import transforms
import os
from PIL import Image
from tqdm import tqdm
import csv
import utils_data
import utils_model
import yaml

def load_config(config_file='config.yaml'):
    with open(config_file, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config

def extract_features_and_generate_labels(frame_store, save_state_dir, save_feature_dir, model, transform, device):
    """
    从视频帧数据中一次推理，提取特征、预测标签，并同时保存：
      - CSV 文件：每一行包含特征向量（例如2048维）、预测标签、实际标签
      - TXT 文件：保存每一帧的预测标签，用于后续比较或其它用途

    参数:
      - frame_store: VideoFrameStore 实例，包含视频帧及其状态。
      - save_dir: 保存生成文件的目录。
      - model: 用于推理的模型，应返回 (feature, outputs) ，其中 feature 用于特征保存，outputs 为分类输出 logits。
      - transform: 图像预处理转换操作。
      - device: 计算设备（CPU 或 GPU）。
    """
    model.eval()
    if not os.path.exists(save_state_dir):
        os.makedirs(save_state_dir)
        
    if not os.path.exists(save_feature_dir):
        os.makedirs(save_feature_dir)

    # 遍历每个视频
    for video_id, frames in frame_store.data.items():
        csv_file = os.path.join(save_feature_dir, f"{video_id}_features.csv")
        txt_file = os.path.join(save_state_dir, f"{video_id}.txt")
        
        correct_predictions = 0
        total_frames = 0

        # 打开 CSV 文件写入标题，同时打开 TXT 文件用于写入预测标签
        with open(csv_file, mode='w', newline='') as csvfile, open(txt_file, mode='w', encoding='utf-8') as txtfile:
            writer = csv.writer(csvfile)
            # 假设特征维度为2048；可以根据实际模型的输出修改此处
            writer.writerow([f"Feature_{i}" for i in range(2048)])
            
            frame_items = list(frames.items())
            loop = tqdm(frame_items, desc=f"Processing {video_id}", unit="frame")
            for frame_number, (frame_path, state) in loop:
                if not frame_path:
                    continue
                # 加载图像并转换
                try:
                    image = Image.open(frame_path).convert("RGB")
                except Exception as e:
                    print(f"加载图像 {frame_path} 出错：{e}")
                    continue
                image = transform(image).unsqueeze(0).to(device)
                
                with torch.no_grad():
                    # model 返回 (feature, outputs)
                    feature, outputs = model(image)
                
                # 将 feature 展平为一维，并转移到 CPU
                feature = feature.view(-1).cpu().numpy()
                # 获取预测标签（互斥分类使用 argmax）
                predicted_label = torch.argmax(outputs, dim=1).item()
                # 更新准确率统计
                if predicted_label == state:
                    correct_predictions += 1
                total_frames += 1

                # 写入 CSV 行：特征、预测标签、实际标签
                writer.writerow(list(feature))
                # 写入 TXT 文件：预测标签（每行一个标签）
                txtfile.write(f"{predicted_label}\n")
                
                loop.set_postfix({"Processed": total_frames})
        
        # 计算并打印视频准确率
        accuracy = correct_predictions / total_frames if total_frames > 0 else 0
        print(f"视频 {video_id} 处理完成，保存 CSV 到 {csv_file}，TXT 到 {txt_file}。准确率: {accuracy:.4f}")

if __name__ == '__main__':
    config = load_config("config/predicate_config.yaml")
    # 示例路径配置
    frame_path = config['paths']['frame_path']
    save_state_dir = config['paths']['save_state_dir']
    save_feature_dir = config['paths']['save_feature_cnn_dir']
    
    # 创建数据集实例（这里假设 utils_data.VideoFrameStore 根据 frame_path 读取视频帧数据，并组织为 {video_id: {frame_number: (frame_path, state)} }）
    frame_store = utils_data.VideoFrameStore(frame_path)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 加载预训练模型（示例中使用 ResNet50V2FeatureExtractor，自行根据实际情况修改）
    model = utils_model.ResNet50V2FeatureExtractor(num_classes=7, pretrained=True)
    model = model.to(device)
    
    # 加载权重文件（可选）
    path_model = config['paths']['path_feature_model']
    state_dict = torch.load(path_model, map_location=device)
    model.load_state_dict(state_dict, strict=False)
    
    # 定义图像转换
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 调用合并函数
    extract_features_and_generate_labels(frame_store, save_state_dir, save_feature_dir, model=model, transform=transform, device=device)
