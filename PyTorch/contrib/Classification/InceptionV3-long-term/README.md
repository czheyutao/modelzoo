
# Inception_v3
## 1. 模型概述
Inception_v3是Google提出的深度卷积神经网络架构，属于 Inception 系列（如 Inception-v1、Inception-v2 等）的改进版本，主要用于图像分类和目标识别任务。它在 ILSVRC 2015中表现优异，显著降低了分类错误率，同时保持了较高的计算效率。

- 论文链接：[[1512.00567v3\]]Rethinking the Inception Architecture for Computer Vision(https://doi.org/10.48550/arXiv.1512.00567)
- 仓库链接：https://github.com/open-mmlab/mmpretrain/tree/main/configs/inception_v3

## 2. 快速开始
使用本模型执行训练的主要流程如下：
1. 基础环境安装：介绍训练前需要完成的基础环境检查和安装。
2. 获取数据集：介绍如何获取训练所需的数据集。
3. 构建环境：介绍如何构建模型运行所需要的环境。
4. 启动训练：介绍如何运行训练。

### 2.1 基础环境安装

请参考基础环境安装章节，完成训练前的基础环境检查和安装。

### 2.2 准备数据集
#### 2.2.1 获取数据集
Inception_v3 使用 ImageNet 数据集，该数据集为开源数据集，可从 [ImageNet](https://image-net.org/) 下载。

#### 2.2.2 处理数据集
具体配置方式可参考：https://blog.csdn.net/xzxg001/article/details/142465729。


### 2.3 构建环境

所使用的环境下已经包含PyTorch框架虚拟环境。
1. 执行以下命令，启动虚拟环境。
    ```
    conda activate torch_env
    ```
2. 安装python依赖。
    ```
    pip3 install  -U openmim 
    pip3 install git+https://gitee.com/xiwei777/mmengine_sdaa.git 
    pip3 install opencv_python mmcv --no-deps
    mim install -e .
    pip install -r requirements.txt

    ```

### 2.4 启动训练

1. 在构建好的环境中，进入训练脚本所在目录。
  ```
  cd <ModelZoo_path>/PyTorch/contrib/Classification/InceptionV3/run_scripts
  ```

2. 运行训练。该模型支持单机单卡。
  ```
  python run_demo.py --config ../configs/inception_v3/inception-v3_8xb32_in1k.py \
      --launcher pytorch --nproc-per-node 4 --amp \
      --cfg-options "train_dataloader.dataset.data_root=$data_path" "val_dataloader.dataset.data_root=$data_path" 2>&1 | tee sdaa.log
  ```
    更多训练参数参考 run_scripts/argument.py
