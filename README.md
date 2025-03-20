# 引入tool Detection和Smooth Bootstrap的Robust Noise

### 文件说明
- tool_detection.py  
  使用CNN对帧进行特征提取，辨别当前使用的工具类型，并保存权重文件
- feature_detection.py  
  使用CNN对帧进行特征提取，辨别当前所属状态，并保存权重文件
- generate_tool.py  
  利用tool_detection.py 生成的权重文件，对所有视频画面进行处理，保存带有噪声的tool分布
- extracate_features.py  
  利用feature_detection.py 生成的权重文件，对所有视频画面进行处理，保存带有噪声的特征分布，同时保存了noisy label，后续使用的noisy label是时序网络产生的
- sbs_detection.py  
  利用干净的tool分布，干净的annotation和带噪的feature进行时间序列网络训练，并且保存训练权重
- sbs_generate_state.py  
  利用sbs_detection.py生成的权重文件对所有视频帧就行预测，并且保留state，作为今后使用的噪声数据集，使用带噪的feature分布和干净的tool分布  
- robust_learning.py  
  使用带噪的feature，带噪的tool分布，带噪的state进行Smooth Bootstrap训练，并在测试集上测试，所有数据集都是没有事先训练的
- CE_learning.py  
  传统Cross Entropy损失函数，学习训练
- GCE_learning.py
  使用GCE损失函数，学习训练  
- data_comp.ipynb  
  用于数据可视化对比等  
- data_preprocess.py  
  进行数据预处理准备
- feature_learning.py  
  仅将2048维度的图片特征作为输入进行学习训练，采用SBS损失函数  
- utils系列
  用于存储必要的函数文件

---

### 文件夹说明
- config  
  保存配置文件
- cnn_noisy_features  
  保存CNN网络生成的2048维度特征  
- log  
  保存训练过程数据
- weight  
  保存训练权重文件
- LSTM_SBS_state  
  保存时间序列网络生成的噪声，也是后期抗噪声训练的噪声数据来源
- noisy_state  
  CNN提取特征顺便保存的噪声数据，是坏噪声
- noisy_tools  
  tool_detection生成的带噪声的tool分布
- splite_lstm  
  时间序列模型训练的训练集合和测试集合
- splite_noise  
  生成噪声数据的来源，在train数据上训练，test数据上生成噪声

---

### 工程文件使用说明  
- data_preprocess.py数据预处理
- tool_detection.py 和 feature_detection.py 可并行运行保存权重文件
- generate_tool.py  和 extracate_features.py  可并行运行，利用保存的权重进行噪声生成
- sbs_detection.py 训练时间序列网络，保存权重文件
- sbs_generate_state.py  利用保存的权重进行状态噪声生成
- robust_learning.py & CE_learning.py & GCE_learning.py & feature_learning.py 可并行运行，从SBS，CE，GCE和single task，共4个方面进行带噪训练
- data_comp.ipynb 数据可视化对比  


#### 工程文件执行顺序  
解压数据集  

``` python
pip install -r requirements.txt
```

- 修改**config/generate_config.py**  
``` python
python config/generate_config.py
```

``` bash
bash run_pipeline.sh
```

---

### 运行环境
    镜像：PyTorch 2.0.0   Python 3.8(ubuntu20.04)    Cuda 11.8
    GPU: RTX 4090(24GB) * 1
    CPU: 16 vCPU Intel(R) Xeon(R) Gold 6430

---
### 小tips
- 噪声生成方式对比（训练中）  
 已经完成了两轮噪声对比，在“糟糕噪声”中正在运行Smooth BootStrap进行对比
 目前来看，效果好于CE，1~2个百分点，效果好于single task 3~4个百分点
- 关于噪声数据  
 feature的噪声和tool的噪声是在**splite_noise**中train，并在test上进行预测，产生的噪声数据
 后期使用**splite_tool**，仅将标注文件更换为**sbs_generate_state.py**生成的带噪标注文件，tool分布和feature噪声依然使用前期生成的。逻辑上可能稍微有一些漏洞，但是在实际使用、算力节约、时间成本上来看，是有利的
- 关于训练参数
 数据文件只需要更换文件夹根目录即可  
 CNN的batch size在并行运行过程中设置为100，预计消耗现存21GB，可在主流旗舰显卡运行  
 

---

### 部分实验进展

- 2025.1 完整运行baseline，保存模型权重，使用数据增强，命名dataaug  
- 2025.2 生成噪声标签，噪声率30%，进行带噪声训练，保存权重，命名noise，由于测试数据和训练数据噪声率相差较大，参考价值不大  
- 2025.2 在训练数据上重新分割训练集合和测试集合，从而保证噪声率近似，训练后保存权重，命名为mini_noise  
- 2025.2 使用mixup噪声处理方法，保存数据、权重，命名为mini_mixup_noise  
- 2025.2.15 使用ITLM噪声处理方法，保存数据、权重，命名为mini_ITLM_noise  
- 2025.3.1 使用全局卷积的方式进行tool detection任务，划分训练集合、测试集合比例为1:7（最大限度保留可用于后续CNN+LSTM）的可用未学习的训练数据。使用ResNet50（pretrained on ImageNet），运行程序：**tool detection.py**, 保留log与CNN weight名称：**tool detection**。  
- 2025.3.2 进行第二次全局卷积的tool detection任务，并在此基础上通过**generate_tool.py**针对全部数据集进行tool detection任务，并保存为**noisy tools**文件。生成用于后续模态融合的tool文件（带噪声）  
- 2025.3.2 划分tool noise生成的训练数据、state noise生成的训练数据，最终LSTM训练数据，保存在splite_noise(trian：用于训练小规模CNN权重， Test：用于最后生成带噪数据集的数据)， splite_tool（trian：用于训练工具识别，test：用于最终生成带噪的工具）  
- 2025.3.2 进行划分数据集后的状态噪声生成权重训练  
- 2025.3.3 所有预处理数据及标签准备完毕，state的噪声率是33%  
- 3.3 创建LSTM_comp文件夹，用于保存单模态和多模态训练的对比
- 3.3 保存带噪数据的噪声2048维度特征，作为LSTM的输入端口
- 3.3~3.5 运行**tool_lstm.py**和**norm_lstm.py**进行对比，程序运行时间预习20h，遭遇2次cuda error。结果：相较于单模态模型，多模态模型在训练拟合准确率低7个百分点，clean data上的准确率高1个百分点。  
- 3.5  生成对比过程可视化，验证用于LSTM模型的数据集噪声率：state噪声率：37%， tool帧噪声率：46.44%，tool噪声率：4.365%。
- 3.5 分析state噪声分布，个人认为通过在少量的数据集上进行训练，并预测剩余数据集制造noise并不合理。通过state分布可以看出，此类噪声具有严重的分配不合理性。文件保存在**data_comp.ipynb** 
- 利用少量clean data进行fine tune
- ITLM：逐步舍弃高loss值的video数据  

- 2025.3.6 开启ITLM类型的noisy robust测试，预计24小时内完成实验  
- 3.7  引入Clean data对模型进行微调，对比Tool annotation引入是否具有效果（实验结果是有效果的）  
- 3.8  学习GCE Loss Function，并使用CIFAR-10/100N进行实验  
- 3.8  将GCE Loss作为Noisy Robust Loss Function引入LSTM训练当中

- extracate_features.py生成的noisy label
  作为噪声生成任务的对比实验
- 引入tool task以及CE函数作为对比实验（未完成）
- 引入tool task以及GCE函数作为对比实验（未完成）
- 仅使用features作为时序网络输入（未完成）
