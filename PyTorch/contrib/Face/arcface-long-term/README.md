# distributed-pytorch

Code for the DDP tutorial series at https://pytorch.org/tutorials/beginner/ddp_series_intro.html
# ArcFace
## 1. 模型概述
ArcFace是一种用于人脸识别的深度学习模型，由Deng等人在2019年提出，它基于经典的softmax损失函数，融合了加性角度边距的思想以提升类别间的判别能力。类似于传统的分类网络，ArcFace通过在特征空间中引入明确的角度边距，使得不同类别的人脸特征在角度上拥有更大间隔，从而增强模型的识别准确性。ArcFace的核心创新在于其加性角度边距损失（Additive Angular Margin Loss），该损失函数具有清晰的几何意义，有助于学习更加紧凑且可分离的特征分布。此外，针对现实场景中标签噪声较多的问题，ArcFace还扩展提出了子中心机制（sub-center），为每个类别设计多个子中心，允许样本靠近任一子中心，有效缓解噪声影响并提升鲁棒性。ArcFace广泛应用于大规模人脸识别任务中，显著提升了识别性能，成为人脸识别领域的标杆方法之一。


## 2. 快速开始
使用本模型执行训练的主要流程如下：
1. 基础环境安装：介绍训练前需要完成的基础环境检查和安装。
2. 获取数据集：介绍如何获取训练所需的数据集。
3. 构建环境：介绍如何构建模型运行所需要的环境
4. 启动训练：介绍如何运行训练。


### 2.1  基础环境安装
请参考基础环境安装章节，完成训练前的基础环境检查和安装。


###2.2 准备数据集
####2.2.1 获取数据集
此处ArcFace使用的是 Inshop 数据集，需要下载数据集。
下载链接为https://mmlab.ie.cuhk.edu.hk/projects/DeepFashion/InShopRetrieval.html

####2.2.2 处理数据集
解压数据集，使用如下格式进行训练
```
|-inshop
    |-Anno
        |-...
    |-Eval
        |-list_eval_partition.txt
    |-Img
        |-img
            |-MEN
                |-Pants
                |-...
            |-WOMEN
```

###2.3 构建环境
所使用的环境下已经包含PyTorch框架虚拟环境

执行以下命令，启动虚拟环境。
conda activate torch_env

安装python依赖
pip install -r requirements.txt


###2.4 启动训练
在构建好的环境中，进入训练脚本所在目录。

cd <ModelZoo_path>/PyTorch/contrib/Face/ArcFace/run_scripts
运行训练。该模型支持单机单卡.

单机单卡
python run_scripts/run_arcface.py \
  --config config/arcface/resnet50-arcface_8xb32_inshop.py \
  --launcher pytorch --nproc-per-node 4 \
  --cfg-options \
    "train_dataloader.dataset.data_root=$data_path" \
    "val_dataloader.dataset.data_root=$data_path" \
    "test_dataloader.dataset.data_root=$data_path" \
    "query_dataloader.dataset.data_root=$data_path" \
    "gallery_dataloader.dataset.data_root=$data_path" \
    "model.prototype.dataset.data_root=$data_path" \
  2>&1 | tee cuda.log

### 2.4 训练结果
sdaa结果如下  
|加速卡数量|模型|混合精度|Epoch|Batch size|Recall@1|mAP@10|
| :-: | :-: | :-: | :-: | :-: | :-: | :-: |
|1|ArcFace|是|50|64|90.2377|70.1729|


cuda结果如下
|加速卡数量|模型|混合精度|Epoch|Batch size|Recall@1|mAP@10|
| :-: | :-: | :-: | :-: | :-: | :-: | :-: |
|1|ArcFace|是|50|64|90.18|69.3|