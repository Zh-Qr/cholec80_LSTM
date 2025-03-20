import torch
import torch.nn as nn
from torchvision import models, transforms
from torchvision.models import ResNet50_Weights
from PIL import Image
import yaml
import os

def load_config(config_file='config.yaml'):
    with open(config_file, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config

config = load_config("config/predicate_config.yaml")
# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 定义类别数
num_classes = 7

# 构建模型：使用预训练的 ResNet50，并将最后一层替换为适合多标签分类的全连接层
model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
in_features = model.fc.in_features
model.fc = nn.Linear(in_features, num_classes)
model = model.to(device)

# 加载模型权重（确保文件路径正确，此处假设权重文件名为 weight/tool_detection.pth）
checkpoint_path = config['paths']['path_tool_model']
state_dict = torch.load(checkpoint_path, map_location=device)
model.load_state_dict(state_dict, strict=False)
model.eval()

# 定义图像预处理（与训练/测试时保持一致）
test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# 输入帧文件夹路径
frames_root = config['paths']['frames_root']
# 输出预测结果文件夹（若不存在则自动创建）
output_dir = config['paths']['save_tool_dir']
os.makedirs(output_dir, exist_ok=True)

# 工具名称列表（顺序与模型输出对应）
tool_names = ["Grasper", "Bipolar", "Hook", "Scissors", "Clipper", "Irrigator", "SpecimenBag"]

# 获取视频列表（每个视频对应一个子文件夹）
video_list = os.listdir(frames_root)

for video in video_list:
    video_dir = os.path.join(frames_root, video)
    if not os.path.isdir(video_dir):
        continue  # 非目录跳过

    # 获取视频中所有帧文件（按文件名排序，可根据实际情况修改排序逻辑）
    frame_files = sorted(os.listdir(video_dir))
    
    # 输出结果文件路径（建议不要使用空格，使用下划线）
    output_file = os.path.join(output_dir, f"{video}.txt")
    
    with open(output_file, 'w') as f:
        # 写入标题行
        header_line = "\t".join(tool_names)
        f.write(header_line + "\n")
        
        # 遍历视频中的每一帧图像
        for frame_file in frame_files:
            frame_full_path = os.path.join(video_dir, frame_file)
            try:
                image = Image.open(frame_full_path).convert("RGB")
            except Exception as e:
                print(f"无法打开图像 {frame_full_path}: {e}")
                continue
            
            image_tensor = test_transform(image).unsqueeze(0)
            with torch.no_grad():
                outputs = model(image_tensor.to(device))
                probabilities = torch.sigmoid(outputs)
                # 阈值 0.5 判断工具是否出现
                predictions = (probabilities > 0.5).int().cpu().numpy()[0]
            
            # 将预测结果转换为制表符分隔的字符串，如 "1\t0\t1\t0\t0\t1\t0"
            prediction_str = "\t".join(str(int(p)) for p in predictions)
            f.write(prediction_str + "\n")
    
    print(f"视频 {video} 的预测结果已保存到 {output_file}")
