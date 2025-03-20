import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
import utils_data
import utils_model
import utils_LSTM

directory = 'autodl-tmp'
save_name = 'LSTM_size100'

# 获取 video_id 列表
def get_videoid(directory):
    video_ids = []
    with open(directory, 'r') as file:
        for line in file:
            video_id = str(line.strip())  
            video_ids.append(video_id)
    return video_ids

if __name__ == '__main__':
    
    train_videoids = get_videoid('splite_dir/train_videos.txt')
    test_videoids = get_videoid('splite_dir/test_videos.txt')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    lstm_model = utils_model.LSTMModel()
    lstm_model = lstm_model.to(device)

    # 选择损失函数（假设分类任务是多类分类）
    criterion = nn.CrossEntropyLoss()

    # 选择优化器，通常使用Adam或SGD
    optimizer = torch.optim.Adam(lstm_model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma = 0.1)
    train_datasets = []
    for video_id in train_videoids:
        train_dataset = utils_data.LSTMDataset(directory, video_id)
        train_datasets.append(train_dataset)

    test_datasets = []
    for video_id in test_videoids:
        test_dataset = utils_data.LSTMDataset(directory, video_id)
        test_datasets.append(test_dataset)
        
    print("数据集加载完成")
    
    train_loaders = [DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4) for train_dataset in train_datasets]
    test_loaders = [DataLoader(test_dataset, batch_size=32, shuffle=True, num_workers=4) for test_dataset in test_datasets]
    
    try:
        utils_LSTM.start_train(num_epochs = 100, save_name = save_name, model = lstm_model, train_loaders = train_loaders, test_loaders = test_loaders, criterion = criterion, optimizer = optimizer, scheduler = scheduler, device = device)
    except KeyboardInterrupt:
        print("训练被中断，正在保存模型权重...")
        torch.save(lstm_model.state_dict(), f"weight/{save_name}")
        print(f"模型权重已保存为 weight/{save_name}")
    