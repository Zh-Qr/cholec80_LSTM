import yaml
from datetime import datetime

# 计算当前时间字符串
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

## 训练参数
cnn_num_epoch = 100
cnn_batch_size = 100
cnn_lr = 0.001
cnn_step_size = 30
cnn_gamma = 0.1

lstm_num_epoch = 100
lstm_batch_size = 128
lstm_lr = 0.001
lstm_step_size = 30
lstm_gamma = 0.1
window_size = 100
param = 5

## 路径
frame_path = "../autodl-tmp"
clean_state_directory = f'{frame_path}/annotations'
raw_state_directory = f'{frame_path}/phase_annotations'
clean_tool_directory = f'{frame_path}/tool'
raw_tool_directory = f'{frame_path}/tool_annotations'
frames_root = f'{frame_path}/frames'
noisy_tools = 'noisy_tools'
noisy_states = 'noisy_states'
cnn_noisy_features = 'cnn_noisy_features'
LSTM_SBS_state = f'LSTM_tool_SBS_state_{timestamp}'

## 权重名（统一在后面加上时间日期）
feature_weight = f'feature_detection_{timestamp}'
tool_weight = f'tool_detection_{timestamp}'
sbs_weight = LSTM_SBS_state

def write_yaml(config, filename):
    with open(filename, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
    print(f"生成 {filename}")

def main():
    # predicate_config.yaml 配置
    predicate_config = {
        "cnn": {
            "num_epochs": cnn_num_epoch,
            "batch_size": cnn_batch_size,
            "lr": cnn_lr,
            "step_size": cnn_step_size,
            "gamma": cnn_gamma
        },
        "lstm": {
            "num_epochs": lstm_num_epoch,
            "batch_size": lstm_batch_size,
            "lr": lstm_lr,
            "step_size": lstm_step_size,
            "gamma": lstm_gamma,
            "window_size": window_size,
            "param": param
        },
        "paths": {
            "frame_path": frame_path,
            "clean_state_directory": clean_state_directory,
            "clean_tool_directory": clean_tool_directory,
            "frames_root": frames_root,
            "save_tool_dir": noisy_tools,
            "save_state_dir": noisy_states,
            "save_feature_cnn_dir": cnn_noisy_features,
            "save_LSTMstate_dir": sbs_weight,
            "path_feature_model": f"weight/{feature_weight}.pth",
            "path_tool_model": f"weight/{tool_weight}.pth",
            "path_LSTM_model": f"weight/{sbs_weight}.pth"
        },
        "prepare_loss": {
            "train_splite_dir": "splite_noise/train_videos.txt",
            "test_splite_dir": "splite_noise/test_videos.txt"
        }
    }

    # preprocess_config.yaml 配置
    preprocess_config = {
        "extract_frame": {
            "ROOT_DIR": frame_path,
            "save_dir": frames_root,
            "label": False
        },
        "process_annotation": {
            "path_annot": raw_state_directory,
            "out_paths": clean_state_directory,
            "label": False
        },
        "process_tool": {
            "file_path": raw_tool_directory,
            "tool_dir": clean_tool_directory,
            "label": False
        },
        "splite_dir": {
            "frame_dir": frames_root,
            "splite_noise": "splite_noise",
            "splite_lstm": "splite_lstm",
            "ration": [10, 58, 12],
            "label": True
        }
    }

    # robust_config.yaml 配置
    robust_config = {
        "cnn": {
            "num_epochs": cnn_num_epoch,
            "batch_size": cnn_batch_size,
            "lr": cnn_lr,
            "step_size": cnn_step_size,
            "gamma": cnn_gamma
        },
        "lstm": {
            "num_epochs": lstm_num_epoch,
            "batch_size": lstm_batch_size,
            "lr": lstm_lr,
            "step_size": lstm_step_size,
            "gamma": lstm_gamma,
            "window_size": window_size,
            "param": param
        },
        "paths": {
            "frame_path": frame_path,
            "clean_state_directory": clean_state_directory,
            "clean_tool_directory": clean_tool_directory,
            "frames_root": frames_root,
            "save_tool_dir": noisy_tools,
            "save_feature_cnn_dir": cnn_noisy_features,
            "save_LSTMstate_dir": LSTM_SBS_state,
            "save_robustLSTM_weight": f"LSTM_SBS{timestamp}",
            "save_LSTMCE_weight": f"LSTM_CE_{timestamp}",
            "save_LSTMGCE_weight": f"LSTM_GCE_{timestamp}",
            "save_notool_LSTM_weight": f"LSTM_notool_SBS_{timestamp}"
        },
        "prepare_loss": {
            "train_splite_dir": "splite_noise/train_videos.txt",
            "test_splite_dir": "splite_noise/test_videos.txt"
        },
        "robust_loss": {
            "train_splite_dir": "splite_lstm/train_videos.txt",
            "test_splite_dir": "splite_lstm/test_videos.txt"
        }
    }

    # train_config.yaml 配置
    train_config = {
        "cnn": {
            "num_epochs": cnn_num_epoch,
            "batch_size": cnn_batch_size,
            "lr": cnn_lr,
            "step_size": cnn_step_size,
            "gamma": cnn_gamma
        },
        "lstm": {
            "num_epochs": lstm_num_epoch,
            "batch_size": lstm_batch_size,
            "lr": lstm_lr,
            "step_size": lstm_step_size,
            "gamma": lstm_gamma,
            "window_size": window_size,
            "param": param
        },
        "paths": {
            "frame_path": frame_path,
            "clean_state_directory": clean_state_directory,
            "clean_tool_directory": clean_tool_directory,
            "frames_root": frames_root,
            "tool_detection": f"{tool_weight}",
            "feature_detection": f"{feature_weight}",
            "save_feature_cnn_dir": cnn_noisy_features,
        },
        "prepare_loss": {
            "train_splite_dir": "splite_noise/train_videos.txt",
            "test_splite_dir": "splite_noise/test_videos.txt"
        }
    }

    # 写入 YAML 文件
    write_yaml(predicate_config, "predicate_config.yaml")
    write_yaml(preprocess_config, "preprocess_config.yaml")
    write_yaml(robust_config, "robust_config.yaml")
    write_yaml(train_config, "train_config.yaml")

if __name__ == '__main__':
    main()
