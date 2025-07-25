export TORCH_SDAA_AUTOLOAD=cuda_migrate

python -m torch.distributed.launch \
    --nproc_per_node=4 \
    train.py \
    --name cifar10_ViT-L_16 \
    --dataset cifar10 \
    --model_type ViT-L_16 \
    --pretrained_dir ./checkpoints/ViT-L_16.npz \
    --train_batch_size 16 \
    --fp16_opt_level 01 \
    --num_epochs 1 \
    --output_dir ./output


python test_model.py \
    --name test_run \
    --dataset cifar10 \
    --model_type ViT-L_16 \
    --checkpoint_path ./output/cifar10_ViT-L_16_checkpoint.bin \
    --output_dir ./output_test \
    --eval_batch_size 64 \
    --fp16 \
    --seed 42

