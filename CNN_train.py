import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.optim as optim
import torch.nn as nn
import utils_data
import utils_model

#########加载全部数据路径############
path_frame = '../autodl-tmp'

# 创建数据集实例
frame_store = utils_data.VideoFrameStore(path_frame)

#############划分数据集##############
split_dir = 'splite_noise'
train_frames, test_frames = utils_data.split_files_and_extract_frames(split_dir, frame_store, train_ratio=0.8)

#############数据加载器##############
# 数据增强和转换
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),  # 降低图像尺寸
    transforms.ToTensor(),  # 转换为Tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # 归一化
    transforms.RandomHorizontalFlip(),  # 随机水平镜像
    transforms.RandomErasing(scale=(0.04, 0.2), ratio=(0.5, 2)),  # 随机遮挡
    transforms.RandomCrop(224, padding=28),  # 随机裁剪并填充
])

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),  # 降低图像尺寸
    transforms.ToTensor(),  # 转换为Tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 归一化
])

train_slides = list(train_frames.keys())
test_slides = list(test_frames.keys())

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = utils_model.ResNet50V2FeatureExtractor(num_classes=7, pretrained=True)
model = model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma = 0.1)

if __name__ == '__main__':
    # 创建数据集实例，传入指定的视频ID（例如train_videos和test_videos）
    train_dataset = utils_data.VideoFrameDataset(train_frames, transform=train_transform)
    test_dataset = utils_data.VideoFrameDataset(test_frames, transform=test_transform)
    
    print("数据集加载完成")

    # 设置DataLoader时的batch_size为16来减小内存占用
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=16)
    print("训练数据加载器完成")
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=16)
    print("测试数据加载器完成")
    save_name="generate_loss"
    try:
        # 开始训练
        utils_model.start_train(num_epochs=50, save_name=save_name, model=model, train_loader=train_loader, test_loader=test_loader, criterion=criterion, optimizer=optimizer, scheduler=scheduler, device=device)
    except KeyboardInterrupt:
        print("训练被中断，正在保存模型权重...")
        torch.save(model.state_dict(), f'weight/{save_name}.pth')
        print(f"模型权重已保存为 {save_name}")