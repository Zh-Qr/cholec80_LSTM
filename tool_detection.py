import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from torchvision.models import ResNet50_Weights
import os
from PIL import Image
import csv
from tqdm import tqdm
import datetime
import yaml
import utils_model
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
# print("env is done")

def load_config(config_file='config.yaml'):
    with open(config_file, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config

config = load_config("config/train_config.yaml")

frame_path = config['paths']['frames_root']
annotations_path = config['paths']['clean_tool_directory']
train_splite_dir = config['prepare_loss']['train_splite_dir']
test_splite_dir = config['prepare_loss']['test_splite_dir']
save_name = config['paths']['tool_detection']

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

class Cholec80Dataset(Dataset):
    def __init__(self, video_list, frame_path, annotation_path, transform=None):
        self.video_list = video_list
        self.frame_path = frame_path
        self.annotation_path = annotation_path
        self.transform = transform
        self.samples = []  # 每个元素为 (图像完整路径, 标签 tensor)
        self._load_annotations()

    def _load_annotations(self):
        for video in self.video_list:
            # 获取视频帧所在文件夹
            video_frame_dir = os.path.join(self.frame_path, video)
            if not os.path.exists(video_frame_dir):
                print(f"视频 {video} 的帧文件夹 {video_frame_dir} 不存在，跳过。")
                continue

            # 读取当前视频帧文件夹下的所有图像文件，并按文件名排序，然后舍去第一个图片
            img_list = sorted(os.listdir(video_frame_dir))[1:]
            if not img_list:
                print(f"视频 {video} 文件夹下没有找到图像文件。")
                continue

            # 打开对应的视频标注文件
            ann_file = os.path.join(self.annotation_path, f"{video}.txt")
            if not os.path.exists(ann_file):
                print(f"视频 {video} 的标注文件 {ann_file} 不存在，跳过。")
                continue

            annotations = []
            with open(ann_file, 'r', encoding='utf-8') as f:
                reader = csv.reader(f, delimiter='\t')
                # 跳过第一行（通常为表头）
                next(reader, None)
                for row in reader:
                    # 假设每行格式为: image_id, label1, label2, ..., labelN
                    if len(row) < 2:
                        continue
                    try:
                        # 将标注部分转换为整数（0/1），并构造 tensor
                        labels = [int(x) for x in row[:]]
                    except Exception as e:
                        print(f"解析标注文件 {ann_file} 时出错：{e}，行内容：{row}")
                        continue
                    annotations.append(torch.tensor(labels, dtype=torch.float))

            if len(img_list) != len(annotations):
                print(f"视频 {video} 中图像数量与标注行数不一致：跳过第一个图片后有 {len(img_list)} 张 vs 跳过表头后有 {len(annotations)} 行，按最小数量匹配。")
            sample_num = min(len(img_list), len(annotations))
            for i in range(sample_num):
                img_full_path = os.path.join(video_frame_dir, img_list[i])
                if os.path.exists(img_full_path):
                    self.samples.append((img_full_path, annotations[i]))
                else:
                    print(f"图像文件 {img_full_path} 不存在，跳过。")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label_vector = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label_vector
    
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = 7  # 根据实际需要调整工具类别数
num_epochs = config['cnn']['num_epochs']
learning_rate = config['cnn']['lr']

# 使用预训练的 ResNet50，并将最后一层替换为多标签分类层
# model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
# in_features = model.fc.in_features
# model.fc = nn.Linear(in_features, num_classes)  # 输出层改为 num_classes
model = utils_model.ResNet50V2FeatureExtractor(num_classes=num_classes, pretrained=True)
model = model.to(device)

# 多标签任务常用 BCEWithLogitsLoss（内部包含 sigmoid 激活）
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=config['cnn']['step_size'], gamma = config['cnn']['gamma'])


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

        # 对输出进行 sigmoid 激活，并使用阈值0.5判断
        predicted = (torch.sigmoid(outputs) > 0.5).float()

        # 计算预测正确的标签总数
        correct += (predicted == targets).sum().item()
        total += targets.numel()  # targets 中所有元素的数量

        # 更新进度条后缀显示
        loop.set_postfix({
            "loss": running_loss / (batch_idx + 1),  # 平均损失
            "accuracy": 100. * correct / total         # 当前准确率
        })

    avg_loss = running_loss / len(train_loader)
    accuracy = 100. * correct / total
    return avg_loss, accuracy


def start_train(num_epochs, save_name, model, train_loader, criterion, optimizer, scheduler, device, train_tactic = None):
    os.makedirs("log",exist_ok=True)
    os.makedirs("weight",exist_ok=True)

    
    for epoch in range(num_epochs):
        train_loss, train_accuracy = train(model, train_loader, criterion, optimizer, epoch, device)

        scheduler.step()


        # 打印每个 epoch 的训练和测试结果
        print(f'Epoch {epoch + 1}/{num_epochs} - '
            f'Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%, ')
            # f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%')
    
        if train_accuracy > 99.9:
            break
    
    torch.save(model.state_dict(), f"weight/{save_name}.pth")
    
if __name__ == '__main__':
    with open(train_splite_dir, 'r', encoding='utf-8') as f:
        train_splite = [line.strip() for line in f if line.strip()]

    with open(test_splite_dir, 'r', encoding='utf-8') as f:
        test_splite = [line.strip() for line in f if line.strip()]

    train_dataset = Cholec80Dataset(
        video_list=train_splite,
        frame_path=frame_path,
        annotation_path=annotations_path,
        transform=train_transform
    )

    trainloader = DataLoader(train_dataset, batch_size=config['cnn']['batch_size'], shuffle=True, num_workers=8)

    try:
        start_train(num_epochs = num_epochs,
                                save_name = save_name,
                                model = model,
                                train_loader = trainloader,
                                criterion = criterion,
                                optimizer = optimizer,
                                scheduler = scheduler,
                                device = device)
    except KeyboardInterrupt:
        print("训练被中断，正在保存模型权重...")
        torch.save(model.state_dict(), f'weight/{save_name}.pth')
        print(f"模型权重已保存为 {save_name}.pth")