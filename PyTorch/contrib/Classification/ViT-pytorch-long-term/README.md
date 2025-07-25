# Vision Transformer

## 1. 模型概述
Vision Transformer（ViT）是2020年提出的一种将Transformer架构直接应用于图像分类任务的模型，旨在打破传统卷积神经网络（CNN）在视觉领域的主导地位。ViT将图像划分为固定大小的图像块（patch），并将其序列化后输入Transformer结构进行建模，从而捕捉全局依赖关系。通过完全基于注意力机制而非卷积操作，ViT展现出在大规模数据集上出色的性能，开创了视觉Transformer发展的新方向。

- 论文链接：[An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929)
- 仓库链接：https://github.com/jeonsworld/ViT-pytorch

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
Vision Transformer 使用 Cifar 数据集，该数据集为开源数据集，可从 http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz 下载。

#### 2.2.2 处理数据集


获得数据集后，可以在[data_utils.py](utils/data_utils.py)中修改路径指向数据集所在父文件夹。

例如训练集修改

 ```
    trainset = datasets.CIFAR10(root="/data/teco-data/cifar10",
                                    train=True,
                                    download=False,
                                    transform=transform_train)
```

测试集修改

```
    testset = datasets.CIFAR10(root="/data/teco-data/cifar10",
                            train=False,
                            download=False,
                            transform=transform_test) if args.local_rank in [-1, 0] else None
```

具体配置方式可参考：[CIFAR-10数据集（介绍、下载读取、可视化显示、另存为图片）_cifar10数据集-CSDN博客](https://blog.csdn.net/qq_40755283/article/details/125209463?ops_request_misc=%7B%22request%5Fid%22%3A%223aab7ab8bf44a13c53ce39786533e422%22%2C%22scm%22%3A%2220140713.130102334..%22%7D&request_id=3aab7ab8bf44a13c53ce39786533e422&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~top_positive~default-1-125209463-null-null.142^v102^pc_search_result_base6&utm_term=Cifar)。
### 2.3 构建环境

所使用的环境下已经包含PyTorch框架虚拟环境。
1. 执行以下命令，启动虚拟环境。
    ```
    conda activate torch_env
    ```
2. 安装python依赖。
    ```
    pip install -r requirements.txt
    ```
3. 下载预训练权重。
```
cd checkpoint
wget https://storage.googleapis.com/vit_models/imagenet21k/ViT-L_16.npz
```
### 2.4 启动训练
1. 在构建好的环境中，进入工作所在目录。
    ```
    cd <ModelZoo_path>/PyTorch/contrib/Classification/ViT-pytorch/
    ```
2. 设置[训练和评估脚本](./scripts/run_vit.sh)中的路径，然后运行训练和评估脚本，该模型支持多卡训练。
    ```
    sh ./scripts/run_vit.sh
   ```

### 2.5 训练结果
原论文的结果如下: 
![](img\figure2.png)

训练结果
训练结果
|      模型       |    数据集      |    sdaa复现结果  (Accuracy)       |     CUDA结果
|:---------------|:--------------:|:-----------------:|:-----------------:|
|ViT-L/16   |CIFAR-10  |   98.75  |  99.42

具体如下图所示
![](img\结果.png)