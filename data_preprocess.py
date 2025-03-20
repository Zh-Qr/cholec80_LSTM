"""
视频帧提取、文件划分、tool分布提取
"""
import os
import pandas as pd
import cv2
import yaml
import random

def load_config(config_file='config.yaml'):
    with open(config_file, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config


# 视频帧提取
def extract_frame(ROOT_DIR, save_dir, VIDEO_NAMES):
    for video_name in VIDEO_NAMES:
        path = os.path.join(ROOT_DIR, "videos", video_name)
        save_dirs = save_dir + video_name.replace('.mp4', '') +'/'
        save_dirs = os.path.join(ROOT_DIR, save_dirs)
        if os.path.exists(save_dirs):
            continue
        print(video_name)
        vidcap = cv2.VideoCapture(path)
        fps = vidcap.get(cv2.CAP_PROP_FPS)
        print("fps", fps)
        if fps != 25:
            print(video_name, 'not at 25fps', fps)
        success=True
        count=0
        
        os.makedirs(save_dirs, exist_ok=True)
        while success is True:
            success,image = vidcap.read()
            if success:
                if count % fps == 0:
                    cv2.imwrite(save_dirs + str(int(count//fps)).zfill(5) + '.png', image)
                count+=1
        vidcap.release()
        cv2.destroyAllWindows()
        print(count)
        
# 处理状态标注文件
def process_annotation(path_annot, out_paths):
    phase_to_int = {
        "Preparation": 0,
        "CalotTriangleDissection": 1,
        "ClippingCutting": 2,
        "GallbladderDissection": 3,
        "GallbladderPackaging": 4,
        "CleaningCoagulation": 5,
        "GallbladderRetraction": 6
    }
    
    def process_annotations(input_file, output_file, step=25):
        # 读取标注文件
        annotations = pd.read_csv(input_file, delim_whitespace=True, header=0)
        # 选择每隔25行的数据，并立即创建一个副本避免SettingWithCopyWarning
        selected_annotations = annotations.iloc[::step].copy()
        # 将阶段描述转换为整数
        selected_annotations['Phase'] = selected_annotations['Phase'].map(phase_to_int)
        # 保存到新的文本文件
        selected_annotations['Phase'].to_csv(output_file, index=False, header=False)
        
    input_files = os.listdir(path_annot)
    for input_file in input_files:
        file_path = os.path.join(path_annot, input_file)
        # 修改输出文件名，移除 "-phase" 部分
        output_filename = input_file.replace('-phase.txt', '.txt')
        out_path = os.path.join(out_paths, output_filename)
        process_annotations(file_path, out_path)
        print(f"Processed: {output_filename}")
        
# 处理工具标注文件
def process_tool(file_path, tool_dir):
    file_paths = os.listdir(file_path)
    if not os.path.exists(tool_dir):
        os.makedirs(tool_dir)
        
    for path in file_paths:
        input_file = os.path.join(file_path, path)
        output_file = os.path.join(tool_dir, path)
        output_file = output_file.replace('-tool', '')
        with open(input_file, 'r') as f:
            lines = [line.strip() for line in f if line.strip()]
        
        processed_lines = []
        for line in lines:
            columns = line.split('\t')
            new_columns = columns[1:]
            processed_line = '\t'.join(new_columns)
            processed_lines.append(processed_line)
            
        with open(output_file, 'w') as f:
            for line in processed_lines:
                f.write(line + '\n')
                
        print("新文件已保存到：", output_file)
        
def splite_file(frame_dir, ration, splite_noise, splite_lstm):
    """
    将 frame_dir 下的文件夹名按比例 [10, 58, 12] 分为三份：
    subset1, subset2, subset3

    - splite_noise:
        train_videos.txt => subset1
        test_videos.txt  => subset2 + subset3
    - splite_lstm:
        train_videos.txt => subset2
        test_videos.txt  => subset3
    """
    # 1. 读取所有文件夹名
    dir_names = os.listdir(frame_dir)
    # 如果需要只处理文件夹，可以根据需要做进一步过滤
    # dir_names = [d for d in dir_names if os.path.isdir(os.path.join(frame_dir, d))]

    # 2. 打乱顺序
    random.shuffle(dir_names)

    # 3. 按照给定比例分割
    total = sum(ration)  # 10 + 58 + 12 = 80
    s1 = int(len(dir_names) * ration[0] / total)  # 子集1大小
    s2 = int(len(dir_names) * ration[1] / total)  # 子集2大小
    # 剩余的作为子集3
    s3 = len(dir_names) - s1 - s2

    subset1 = dir_names[:s1]
    subset2 = dir_names[s1:s1 + s2]
    subset3 = dir_names[s1 + s2:]

    # 4. 组合对应的训练/测试集
    noise_train = subset1
    noise_test = subset2 + subset3

    lstm_train = subset2
    lstm_test = subset3

    # 5. 创建输出文件夹并写入
    os.makedirs(splite_noise, exist_ok=True)
    os.makedirs(splite_lstm, exist_ok=True)

    # splite_noise
    with open(os.path.join(splite_noise, "train_videos.txt"), "w") as f:
        for video in noise_train:
            f.write(video + "\n")

    with open(os.path.join(splite_noise, "test_videos.txt"), "w") as f:
        for video in noise_test:
            f.write(video + "\n")

    # splite_lstm
    with open(os.path.join(splite_lstm, "train_videos.txt"), "w") as f:
        for video in lstm_train:
            f.write(video + "\n")

    with open(os.path.join(splite_lstm, "test_videos.txt"), "w") as f:
        for video in lstm_test:
            f.write(video + "\n")

    print("文件已按比例分割并写入到:")
    print(f"  - {splite_noise}/train_videos.txt (子集1)")
    print(f"  - {splite_noise}/test_videos.txt  (子集2+3)")
    print(f"  - {splite_lstm}/train_videos.txt  (子集2)")
    print(f"  - {splite_lstm}/test_videos.txt   (子集3)")
    
        
if __name__ == '__main__':
    config = load_config("config/preprocess_config.yaml")
    
    if config['extract_frame']['label']:
        VIDEO_NAMES = os.listdir(os.path.join(config['extract_frame']['ROOT_DIR'], "videos"))
        extract_frame(config['extract_frame']['ROOT_DIR'], config['extract_frame']['save_dir'], VIDEO_NAMES)
        print(f"视频帧已经提取完毕，保存在{config['save_dir']['ROOT_DIR']}")
        
    if config['process_annotation']['label']:
        process_annotation(config['process_annotation']['path_annot'], config['process_annotation']['out_paths'])
        print(f"状态标注处理完毕，保存在{config['process_annotation']['out_paths']}")
        
    if config['process_tool']['label']:
        process_tool(config['process_tool']['file_path'], config['process_tool']['tool_dir'])
        print(f"tool分布已处理完毕，保存在{config['process_tool']['tool_dir']}")
    
    if config['splite_dir']['label']:
        splite_file(config['splite_dir']['frame_dir'], config['splite_dir']['ration'], config['splite_dir']['splite_noise'], config['splite_dir']['splite_lstm'])
    print("数据预处理完毕")