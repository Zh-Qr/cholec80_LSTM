import torch
from torchvision import transforms
import os
from PIL import Image
from collections import deque
import utils_model
import utils_data

directory = '../splite_dir/test_videos.txt'
CNN_model_path = '../weight/tool_aug.pth'
LSTM_model_path = '../weight/LSTM_base'

def predicate_all(directory, CNN_path, LSTM_path):
    with open(directory, "r") as f:
        datas = f.read().splitlines()
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    CNN_model, lstm_model = utils_model.load_model(device, CNN_path, LSTM_path, pretrained=True)
    
    CNN_model.eval()
    lstm_model.eval()

    # 数据转换定义
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225])
    ])

    # 设置 window_size
    window_size = 10

    # 使用 deque 只保存最近 window_size 帧的特征（组合后的特征在 CPU 上保存）
    feature_buffer = deque(maxlen=window_size)

    acc = []

    for video_id in datas:
        data_path = os.path.join('../autodl-tmp/frames', video_id)
        annotation_path = os.path.join('../autodl-tmp/annotations', f'{video_id}.txt')
        video_paths = os.listdir(data_path)

        labels = []
        states = []
        with open(annotation_path, 'r') as file:
            for line in file:
                label = int(line.strip())
                labels.append(label)

        frames = []
        for video_path in video_paths:
            video_path = os.path.join(data_path, video_path)
            frame = Image.open(video_path).convert('RGB')
            frames.append(frame)
        with torch.no_grad():
            for i in range(len(frames)):
                # 读取第 i 帧图像并进行预处理
                image = frames[i]
                image_tensor = transform(image).unsqueeze(0).to(device)  # 形状 [1, 3, 224, 224]

                # 通过 CNN 提取特征和预测 logits
                feature, outputs = CNN_model(image_tensor)

                # 移除 batch 维度（假设 feature 形状为 [1, d]）
                feature = feature.squeeze(0)   # 形状 [d]
                outputs = outputs.squeeze(0)   # 形状 [num_classes]

                # CNN 预测类别：取 logits 在 dim=0 得到最大值对应的类别索引
                _, cnn_predicted_class = torch.max(outputs, dim=0)  # cnn_predicted_class 为标量

                # 将预测的类别转换为 float 类型，并转换为形状 [1] 的一维张量
                predicted_class_tensor = cnn_predicted_class.float().unsqueeze(0)

                # 构造“组合特征”：将 CNN 提取的 feature 与预测类别信息拼接在一起
                combined_feature = torch.cat((feature, predicted_class_tensor, predicted_class_tensor), dim=0)
                # combined_feature 的形状为 [d+2]，是一维向量

                # 将组合特征存入固定长度的滑动窗口中，并确保存到 CPU 上
                feature_buffer.append(combined_feature.cpu())

                if i < window_size:
                    # 前 window_size 帧，直接使用 CNN 的预测结果
                    state_prediction = cnn_predicted_class.item()
                else:
                    # 从第 window_size 帧开始，使用前 10 帧组成的滑动窗口进行 LSTM 预测
                    # 从 feature_buffer 中取出数据，形状为 [window_size, d+2]
                    window = torch.stack(list(feature_buffer), dim=0)
                    # LSTM 期望输入 shape 为 [batch_size, seq_len, input_size]，这里 batch_size=1
                    window = window.unsqueeze(0).to(device)  # 将窗口传回 GPU，形状 [1, window_size, d+2]

                    # 通过 LSTM 进行预测
                    result = lstm_model(window)  # 假设结果形状为 [1, num_classes]
                    _, predicted_state = torch.max(result, dim=1)
                    state_prediction = predicted_state.item()

                states.append(state_prediction)

                # 可选：每处理一定帧数后清空 GPU 缓存
                if i % 50 == 0:
                    torch.cuda.empty_cache()
            sing_accuracy = utils_data.get_accuracy(labels, states)
            acc.append(sing_accuracy)
            print(f"{video_id}处理完毕， 准确率是{sing_accuracy}")
            
    avegerage_acc = sum(acc)/len(acc)
    return avegerage_acc

if __name__ == '__main__':
    avegerage_acc = predicate_all(directory, CNN_model_path, LSTM_model_path)
    print(f"平均准确率是：{avegerage_acc}")
