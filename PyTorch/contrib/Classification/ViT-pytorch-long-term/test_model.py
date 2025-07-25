# coding=utf-8
from __future__ import absolute_import, division, print_function

import argparse
import logging
import os
import random
import numpy as np
from datetime import timedelta

import torch
import torch.distributed as dist
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from torch.sdaa import amp
from torch.nn.parallel import DistributedDataParallel as DDP

from models.modeling import VisionTransformer, CONFIGS
from utils.data_utils import get_loader
from utils.dist_util import get_world_size
from torch_sdaa.utils import cuda_migrate

logger = logging.getLogger(__name__)
from tcap_dllogger import Logger, StdOutBackend, JSONStreamBackend, Verbosity

# 配置日志
json_logger = Logger(
    [
        StdOutBackend(Verbosity.DEFAULT),
        JSONStreamBackend(Verbosity.VERBOSE, 'test_dlloger.json'),
    ]
)
json_logger.metadata("test.loss", {"unit": "", "GOAL": "MINIMIZE", "STAGE": "TEST"})
json_logger.metadata("test.accuracy", {"unit": "%", "GOAL": "MAXIMIZE", "STAGE": "TEST"})

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def simple_accuracy(preds, labels):
    return (preds == labels).mean()

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

def setup(args):
    # Prepare model
    config = CONFIGS[args.model_type]
    num_classes = 10 if args.dataset == "cifar10" else 100

    model = VisionTransformer(config, args.img_size, zero_head=True, num_classes=num_classes)
    
    # 加载预训练权重（如果提供）
    if args.pretrained_dir and os.path.exists(args.pretrained_dir):
        model.load_from(np.load(args.pretrained_dir))
        logger.info("Loaded pretrained weights from %s", args.pretrained_dir)
    
    # 加载检查点权重
    if args.checkpoint_path and os.path.exists(args.checkpoint_path):
        checkpoint = torch.load(args.checkpoint_path, map_location=args.device)
        model.load_state_dict(checkpoint)
        logger.info("Loaded checkpoint from %s", args.checkpoint_path)
    else:
        raise FileNotFoundError(f"Checkpoint file {args.checkpoint_path} not found")

    model.to(args.device)
    num_params = count_parameters(model)

    logger.info("Model configuration: %s", config)
    logger.info("Test parameters: %s", args)
    logger.info("Total Parameters: \t%2.1fM" % num_params)
    return args, model

def count_parameters(model):
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return params / 1000000

def validate(args, model, writer, test_loader):
    # Validation
    eval_losses = AverageMeter()
    logger.info("***** Running Validation *****")
    logger.info("  Num steps = %d", len(test_loader))
    logger.info("  Batch size = %d", args.eval_batch_size)

    model.eval()
    all_preds, all_label = [], []
    epoch_iterator = tqdm(test_loader,
                          desc="Validating... (loss=X.X)",
                          bar_format="{l_bar}{r_bar}",
                          dynamic_ncols=True,
                          disable=args.local_rank not in [-1, 0])
    loss_fct = torch.nn.CrossEntropyLoss()
    for step, batch in enumerate(epoch_iterator):
        batch = tuple(t.to(args.device) for t in batch)
        x, y = batch
        with torch.no_grad():
            logits = model(x)[0]
            eval_loss = loss_fct(logits, y)
            eval_losses.update(eval_loss.item())
            preds = torch.argmax(logits, dim=-1)

        if len(all_preds) == 0:
            all_preds.append(preds.detach().cpu().numpy())
            all_label.append(y.detach().cpu().numpy())
        else:
            all_preds[0] = np.append(all_preds[0], preds.detach().cpu().numpy(), axis=0)
            all_label[0] = np.append(all_label[0], y.detach().cpu().numpy(), axis=0)
        epoch_iterator.set_description("Validating... (loss=%2.5f)" % eval_losses.val)

    all_preds, all_label = all_preds[0], all_label[0]
    accuracy = simple_accuracy(all_preds, all_label)

    logger.info("\n")
    logger.info("Validation Results")
    logger.info("Valid Loss: %2.5f" % eval_losses.avg)
    logger.info("Valid Accuracy: %2.5f" % accuracy)

    if args.local_rank in [-1, 0]:
        writer.add_scalar("test/loss", scalar_value=eval_losses.avg, global_step=0)
        writer.add_scalar("test/accuracy", scalar_value=accuracy, global_step=0)
        json_logger.log(
            step=(0, 0),
            data={
                "rank": os.environ.get("LOCAL_RANK", "0"),
                "test.loss": eval_losses.avg,
                "test.accuracy": accuracy,
            },
            verbosity=Verbosity.DEFAULT,
        )

    return eval_losses.avg, accuracy

def main():
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument("--name", required=True, help="Name of this run. Used for monitoring.")
    parser.add_argument("--dataset", choices=["cifar10", "cifar100"], default="cifar10", help="Which downstream task.")
    parser.add_argument("--model_type", choices=["ViT-B_16", "ViT-B_32", "ViT-L_16", "ViT-L_32", "ViT-H_14", "R50-ViT-B_16"],
                        default="ViT-B_16", help="Which variant to use.")
    parser.add_argument("--pretrained_dir", type=str, default="checkpoint/ViT-B_16.npz",
                        help="Where to search for pretrained ViT models (optional).")
    parser.add_argument("--checkpoint_path", type=str, required=True,
                        help="Path to the checkpoint file to test (.bin file).")
    parser.add_argument("--output_dir", default="output", type=str,
                        help="The output directory where logs will be written.")
    parser.add_argument("--img_size", default=224, type=int, help="Resolution size")
    parser.add_argument("--train_batch_size", default=64, type=int, help="Total batch size for eval.")
    parser.add_argument("--eval_batch_size", default=64, type=int, help="Total batch size for eval.")
    parser.add_argument("--local_rank", type=int, default=-1, help="Local rank for distributed training on GPUs")
    parser.add_argument('--seed', type=int, default=42, help="Random seed for initialization")
    parser.add_argument('--fp16', action='store_true', help="Whether to use 16-bit float precision")
    parser.add_argument('--fp16_opt_level', type=str, default='O2',
                        help="Apex AMP optimization level: ['O0', 'O1', 'O2', 'O3'].")
    
    args = parser.parse_args()

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='tccl', timeout=timedelta(minutes=60))
        args.n_gpu = 1
    args.device = device

    # Setup logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits testing: %s" %
                   (args.local_rank, args.device, args.n_gpu, bool(args.local_rank != -1), args.fp16))

    # Set seed
    set_seed(args)

    # Model & Data Loader Setup
    args, model = setup(args)

    # Distributed training
    if args.local_rank != -1:
        model = DDP(model)

    # Setup TensorBoard
    if args.local_rank in [-1, 0]:
        os.makedirs(args.output_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=os.path.join("logs", args.name))
    else:
        writer = None

    # Load test dataset
    _, test_loader = get_loader(args)

    # Run validation
    loss, accuracy = validate(args, model, writer, test_loader)

    # Clean up
    if args.local_rank in [-1, 0]:
        writer.close()
    logger.info("Test completed. Final Loss: %.5f, Final Accuracy: %.5f", loss, accuracy)

if __name__ == "__main__":
    main()