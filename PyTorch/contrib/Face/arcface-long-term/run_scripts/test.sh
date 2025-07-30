#!/bin/bash
script_path=$(dirname $(readlink -f "$0"))
echo "当前脚本路径: $script_path"

data_path="./data/inshop"

#安装依赖
cd .. 
pip3 install  -U openmim 
pip3 install git+https://gitee.com/xiwei777/mmengine_sdaa.git 
pip3 install opencv_python mmcv --no-deps
mim install -e .
pip install -r requirements.txt
pip3 install numpy==1.24.3

cd "$script_path/.."

#执行训练
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