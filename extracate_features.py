import torch
from torchvision import transforms
import os
from PIL import Image
from tqdm import tqdm
import csv
import utils_data
import utils_model


path_model = 'weight/dataaug'
path_frame = 'autodl-tmp'
save_dir="extracted_features_csv"

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


if __name__ == '__main__':
    # 创建数据集实例
    frame_store = utils_data.VideoFrameStore(path_frame)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_pre = utils_model.ResNet50V2FeatureExtractor(num_classes=7, pretrained=True)
    model_pre = model_pre.to(device)

    state_dict = torch.load(path_model)
    model_pre.load_state_dict(state_dict, strict=False)
    # 数据转换定义
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 调用特征提取函数
    extract_features_with_accuracy(frame_store, save_dir, model=model_pre, transform=transform, device=device)