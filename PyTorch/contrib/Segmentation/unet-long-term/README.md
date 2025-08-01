# Unet
## 1. 模型概述
Unet是一种用于图像分割的卷积神经网络模型，由Olaf Ronneberger等人在2015年提出，它借鉴了VGG等模型的深度结构特点。类似于VGG的统一架构，Unet的编码器部分（收缩路径）使用堆叠的3x3小型卷积核层来提取特征，并通过多个最大池化层进行降采样，逐步增加网络深度并减少特征图尺寸。然而，Unet引入了独特的对称U形设计：在编码器之后，解码器路径（扩展路径）通过上采样操作（如转置卷积）逐步恢复空间分辨率，同时利用跳跃连接将编码器的高分辨率特征图直接融合到解码器中，以保留细节并提升分割精度。这种架构特别适用于像素级预测任务，如生物医学图像分割，其中编码器部分常基于VGG骨干网络实现高效的特征提取。


## 2. 快速开始
使用本模型执行训练的主要流程如下：
1. 基础环境安装：介绍训练前需要完成的基础环境检查和安装。
2. 获取数据集：介绍如何获取训练所需的数据集。
3. 构建环境：介绍如何构建模型运行所需要的环境
4. 启动训练：介绍如何运行训练。


### 2.1  基础环境安装
请参考基础环境安装章节，完成训练前的基础环境检查和安装。


### 2.2 准备数据集
#### 2.2.1 获取数据集
此处unet使用的是CHASE_DB1数据集，需要下载数据集。
下载链接为https://staffnet.kingston.ac.uk/~ku15565/CHASE_DB1/assets/CHASEDB1.zip。

#### 2.2.2 处理数据集
运行如下命令将 CHASE DB1 数据集转换为 MMSegmentation 格式，该脚本将自动建立目录结构：
```python
python tools/dataset_converters/chase_db1.py /path/to/CHASEDB1.zip
```
之后进行解压数据集，运行如下命令：
```bash
unzip CHASEDB1.zip
``` 
解压后的文件将保存在unet-long-term/data/CHASE_DB1中。

### 2.3 构建环境
所使用的环境下已经包含PyTorch框架虚拟环境

执行以下命令，启动虚拟环境。
```
conda activate torch_env
```
```
安装python依赖
pip install -r requirements.txt
```

### 2.4 启动训练
在构建好的环境中，进入训练脚本所在目录。
```
cd <ModelZoo_path>/PyTorch/contrib/Classification/unet-long-term/run_scripts
```
运行训练。该模型支持单机单卡.

单机单卡
```bash
python run_unet.py --config ../configs/unet/unet_s5-d16_deeplabv3_4xb4-40k_chase-db1-128x128.py \
    --launcher pytorch --nproc-per-node 4 --amp \
    --cfg-options "train_dataloader.dataset.dataset.data_root=$data_path" "val_dataloader.dataset.data_root=$data_path" 2>&1 | tee sdaa.log
```

### 2.4 训练结果
|模型         |混合精度  |iter  |Batch size |sdaa mDice    |cuda mDice |
|:----------:|:--------:|:-----:|:---------:|:------------:|:---------:|
|unet|是        |40000      |4          |89.47         |89.49      |