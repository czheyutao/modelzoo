export TORCH_SDAA_AUTOLOAD=cuda_migrate

python test_model.py \
    --name test_run \
    --dataset cifar10 \
    --model_type ViT-L_16 \
    --checkpoint_path /data/suda-data/syc/ViT-pytorch/output/cifar10_ViT-L_16_checkpoint.bin \
    --output_dir output_test \
    --eval_batch_size 64 \
    --fp16 \
    --seed 42
