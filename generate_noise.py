import torch
from torchvision import transforms
import os
from PIL import Image
from tqdm import tqdm
import utils_data
import utils_model

path_frame = 'autodl-tmp'
noisy_file_dir = 'noisy_label'
clean_file_dir = 'autodl-tmp/annotations'
path_model = 'weight/generate_noise'

# 创建数据集实例
frame_store = utils_data.VideoFrameStore(path_frame)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#加载模型
model_pre = utils_model.ResNet50V2FeatureExtractor(num_classes=7, pretrained=True)
model_pre = model_pre.to(device)

state_dict = torch.load(path_model)
model_pre.load_state_dict(state_dict, strict=False)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def generate_noisylabel(frame_store, save_dir, model, transform, device):
    model.eval()
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    for video_id, frames in frame_store.data.items():
        noisy_label_file = os.path.join(save_dir, f"{video_id}.txt")
        
        with open(noisy_label_file, mode='w', newline='', encoding='utf-8') as file:
            frame_items = list(frames.items())
            loop = tqdm(frame_items, desc=f"Extracting features for {video_id}", unit="frame")
            
            for frame_number, (frame_path, state) in loop:
                if frame_path:
                    # 加载并转换图像
                    image = Image.open(frame_path).convert("RGB")
                    image = transform(image).unsqueeze(0).to(device)
                    
                    with torch.no_grad():
                        output = model(image)
                        
                    _, predicted_label = torch.max(output, 1)
                    # 提取tensor中的数值并写入文件
                    file.write(f"{predicted_label.item()}\n")
                    
def get_noisy_rate(clean_file_dir, noisy_file_dir):
    noisy_files = os.listdir(noisy_file_dir)
    total_frames = 0
    noise_frames = 0
    for noisy_filename in noisy_files:
        noisy_file_path = os.path.join(noisy_file_dir, noisy_filename)
        clean_file_path = os.path.join(clean_file_dir, noisy_filename)

        # 检查路径是否指向文件
        if os.path.isfile(noisy_file_path) and os.path.isfile(clean_file_path):
            with open(noisy_file_path, 'r', encoding='utf-8') as noisy_file, \
                open(clean_file_path, 'r', encoding='utf-8') as clean_file:
                noisy_labels = noisy_file.readlines()
                clean_labels = clean_file.readlines()
                
                # 比较两个列表的长度和具体内容
                min_length = min(len(noisy_labels), len(clean_labels))
                total_frames += min_length
                for i in range(min_length):
                    if noisy_labels[i].strip() != clean_labels[i].strip():
                        noise_frames += 1
                        
    noise_ratio = noise_frames / total_frames
    return total_frames, noise_frames, noise_ratio
    
    
                

if __name__ == '__main__':
    #生成噪声标签
    generate_noisylabel(frame_store, save_dir=noisy_file_dir, model=model_pre, transform=transform, device = device)

    #测试噪声率
    total_frames, noise_frames, noise_ratio = get_noisy_rate(clean_file_dir, noisy_file_dir)

    if total_frames > 0:
        print(f"Total frames: {total_frames}, Noise frames: {noise_frames}, Noise ratio: {noise_ratio:.2f}")
    else:
        print("No frames to compare.")