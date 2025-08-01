from __future__ import print_function
import sys

import argparse
import os
import shutil
import time
import random
import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim as optim
import torch.utils.data as data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import pnasnet as pnasnet
import crossentropy
from flops_counter import get_model_complexity_info
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

from utils import Bar, Logger, AverageMeter, accuracy, mkdir_p


import tcap_dllogger
from tcap_dllogger import Logger as Loggerx, StdOutBackend, JSONStreamBackend, Verbosity
json_logger = Loggerx(
    [
        StdOutBackend(Verbosity.DEFAULT),
        JSONStreamBackend(Verbosity.VERBOSE, "dlloger_example.json"),
    ]
)
json_logger.metadata("train.loss", {"unit": "", "GOAL": "MINIMIZE", "STAGE": "TRAIN"})
json_logger.metadata("train.ips",{"unit": "imgs/s", "format": ":.4f", "GOAL": "MAXIMIZE", "STAGE": "TRAIN"})



try:
    import torch_sdaa
    import torch_sdaa.amp as amp 
    use_device = torch.sdaa.is_available()

except:
    print('import torch_sdaa failed ,use gpu')
    import torch.cuda.amp as amp 
    use_device = torch.cuda.is_available()
    cudnn.benchmark = True

import warnings
warnings.filterwarnings('ignore')


# for servers to immediately record the logs
def flush_print(func):
    def new_print(*args, **kwargs):
        func(*args, **kwargs)
        sys.stdout.flush()
    return new_print
p0rint = flush_print(print)

def to_python_float(t):
    return t.item() if isinstance(t, torch.Tensor) else t

from torch.optim.optimizer import Optimizer, required

class LSGD(Optimizer):
    def __init__(self, params, lr=required, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super(LSGD, self).__init__(params, defaults)
        self.print_flag = False  # 添加实例变量

    def __setstate__(self, state):
        super(LSGD, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']


            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data


                if weight_decay != 0:
                    d_p.add_(weight_decay, p.data)

                sz = p.data.size()
                if d_p.dim() == 4 and sz[1] != 1: # we do not consider dw conv
                    assert(weight_decay == 0)
                    sz = p.data.size()
                    w  = p.data.view(sz[0], -1)
                    wstd = w.std(dim=1).view(sz[0], 1, 1, 1)
                    wmean = w.mean(dim=1).view(sz[0], 1, 1, 1)

                    if args.local_rank == 0 and self.print_flag:
                        wm = wstd.view(-1).mean().item()
                        wmm = wmean.view(-1).mean().item()
                        print('lam = %.6f' % args.lam, 'mineps = %.6f' % args.mineps, 
                                '1 - eps/std = %.10f' % (1 - args.mineps / wm), 
                                'std = %.10f' % wm, 'mean = %.10f' % wmm,  'sz = ', sz)
                    
                    d_p.add_(args.lam, (1 - args.mineps / wstd) * (p.data - wmean) + wmean)


                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.zeros_like(p.data)
                        buf.mul_(momentum).add_(d_p)
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(1 - dampening, d_p)
                    if nesterov:
                        d_p = d_p.add(momentum, buf)
                    else:
                        d_p = buf

                p.data.add_(-group['lr'], d_p)

        return loss

# Parse arguments
parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')

parser.add_argument('--cutmix', dest='cutmix', action='store_true')
parser.add_argument('--cutmix_prob', default=1., type=float)

parser.add_argument('--cutout', dest='cutout', action='store_true')
parser.add_argument('--cutout_size', default=112, type=float)

parser.add_argument('--el2', dest='el2', action='store_true', help='whether to use e-shifted L2 regularizer')
parser.add_argument('--mineps', dest='mineps', default=1e-3, type=float, help='min of weights std, typically 1e-3, 1e-8, 1e-2')
parser.add_argument('--lam', dest='lam', default=1e-4, type=float, help='lam of weights for e-shifted L2 regularizer')


parser.add_argument('--nowd-bn', dest='nowd_bn', action='store_true',
                    help='no weight decay on bn weights')
parser.add_argument('--nowd-fc', dest='nowd_fc', action='store_true',
                    help='no weight decay on fc weights')
parser.add_argument('--nowd-conv', dest='nowd_conv', action='store_true',
                    help='no weight decay on conv weights')

# Datasets
parser.add_argument('-d', '--data', default='path to dataset', type=str)
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
# Optimization options
parser.add_argument('--opt-level', default='O1', type=str, 
        help='opt_level must be O0 or O1 for torch.cuda.amp')
parser.add_argument('--keep-batchnorm-fp32', default=True, action='store_true',
                    help='keeping cudnn bn leads to fast training')
parser.add_argument('--loss-scale', type=float, default=None)

parser.add_argument('--label-smoothing', '--ls', default=0.1, type=float)

parser.add_argument('--device', default='cuda', type=str, choices=['cuda', 'sdaa'],
                    help='Specify device type (default: auto-detect)')

parser.add_argument('--mixup', dest='mixup', action='store_true',
                    help='whether to use mixup')
parser.add_argument('--alpha', default=0.2, type=float,
                    metavar='mixup alpha', help='alpha value for mixup B(alpha, alpha) distribution')
parser.add_argument('--cos', dest='cos', action='store_true', 
                    help='using cosine decay lr schedule')
parser.add_argument('--warmup', '--wp', default=5, type=int,
                    help='number of epochs to warmup')
parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--train-batch', default=256, type=int, metavar='N',
                    help='train batchsize (default: 256)')
parser.add_argument('--test-batch', default=200, type=int, metavar='N',
                    help='test batchsize (default: 200)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--drop', '--dropout', default=0, type=float,
                    metavar='Dropout', help='Dropout ratio')
parser.add_argument('--schedule', type=int, nargs='+', default=[30, 60, 90],
                        help='Decrease learning rate at these epochs.')
parser.add_argument('--gamma', type=float, default=0.1, help='LR is multiplied by gamma on schedule.')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--wd-all', dest = 'wdall', action='store_true', 
                    help='weight decay on all parameters')
parser.add_argument('--world_size', default=8, type=int, help="(default:8)")

# Checkpoints
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('-c', '--checkpoint', default='checkpoint', type=str, metavar='PATH',
                    help='path to save checkpoint (default: checkpoint)')
parser.add_argument('-log', default='log.txt', type=str, metavar='PATH',
                    help='log name to save log (default: log.txt)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')

# Architecture
parser.add_argument('--depth', type=int, default=29, help='Model depth.')
parser.add_argument('--cardinality', type=int, default=32, help='ResNet cardinality (group).')
parser.add_argument('--base-width', type=int, default=4, help='ResNet base width.')
parser.add_argument('--widen-factor', type=int, default=4, help='Widen factor. 4 -> 64, 8 -> 128, ...')
# Miscs
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')

parser.add_argument('--use_aux', default=False, action='store_true', help='use auxiliary')
# Device options
# parser.add_argument('--gpu-id', default='0', type=str, help='id(s) for CUDA_VISIBLE_DEVICES')
parser.add_argument('--local_rank', default=0, type=int)
parser.add_argument('--loss_scale', default=128)
parser.add_argument('--max_step', default=0, type=int)


args = parser.parse_args()
state = {k: v for k, v in args._get_kwargs()}

print("opt_level = {}".format(args.opt_level))
print("keep_batchnorm_fp32 = {}".format(args.keep_batchnorm_fp32), type(args.keep_batchnorm_fp32))
print("loss_scale = {}".format(args.loss_scale), type(args.loss_scale))

# Use npu
# os.environ['npu_VISIBLE_DEVICES'] = args.gpu_id

# Random seed
seed=777
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

 
if torch.cuda.is_available():
    print("CUDA is available. Training on GPU.")
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 如果使用多GPU

elif torch.sdaa.is_available():
    print("CUDA is available. Training on SDAA.")
    torch.sdaa.manual_seed(seed)
    torch.sdaa.manual_seed_all(seed)    

best_acc = 0  # best test accuracy

adjusted_lr = args.lr # just for print

def fast_collate(batch):
    imgs = [img[0] for img in batch]
    targets = torch.tensor([target[1] for target in batch], dtype=torch.int64)
    w = imgs[0].size[0]
    h = imgs[0].size[1]
    tensor = torch.zeros((len(imgs), 3, h, w), dtype=torch.uint8)
    for i, img in enumerate(imgs):
        nump_array = np.asarray(img, dtype=np.uint8)
        # tens = torch.from_numpy(nump_array)
        if nump_array.ndim < 3:
            nump_array = np.expand_dims(nump_array, axis=-1)
        nump_array = np.rollaxis(nump_array, 2)

        tensor[i] += torch.from_numpy(nump_array)

    return tensor, targets

class data_prefetcher():
    def __init__(self, loader):
        self.loader = iter(loader)
        if args.device == 'sdaa':
            self.stream = torch.sdaa.Stream()  # SDAA流
            to_device = lambda x: x.sdaa()  # 定义SDAA设备转移函数
        elif args.device == 'cuda':
            self.stream = torch.cuda.Stream()  # CUDA流
            to_device = lambda x: x.cuda()  # 定义CUDA设备转移函数
        self.mean = to_device(
                    torch.tensor([0.5 * 255, 0.5 * 255, 0.5 * 255]).view(1, 3, 1, 1))
        self.std = to_device(
                    torch.tensor([0.5 * 255, 0.5 * 255, 0.5 * 255]).view(1, 3, 1, 1))

        self.preload()

    def preload(self):
        try:
            self.next_input, self.next_target = next(self.loader)
        except StopIteration:
            self.next_input = None
            self.next_target = None
            return
        if self.stream is not None:
            if args.device == 'sdaa':
                stream_context = torch.sdaa.stream(self.stream)
                to_device = lambda x: x.sdaa(non_blocking=True)
            else:  # CUDA
                stream_context = torch.cuda.stream(self.stream)
                to_device = lambda x: x.cuda(non_blocking=True)
            
            with stream_context:
                self.next_input = to_device(self.next_input)
                self.next_target = to_device(self.next_target)
                self.next_input = self.next_input.float()
                self.next_input = self.next_input.sub_(self.mean).div_(self.std)      
             
    def next(self):
        # 仅在支持流的设备上执行同步操作
        if self.stream is not None:
            if args.device == 'sdaa':
                current_stream = torch.sdaa.current_stream()
            else:  # CUDA
                current_stream = torch.cuda.current_stream()
            
            current_stream.wait_stream(self.stream)
        
        input = self.next_input
        target = self.next_target
        
        if input is not None:
            self.preload()
        
        return input, target

def print_func(inputs, prefix):
    if isinstance(inputs, tuple):
        for i in inputs:
            print_func(i, prefix)
    elif isinstance(inputs, torch.Tensor):
        print(prefix, inputs.shape, inputs.dtype)
    else:
        print(prefix, inputs)


def hook_func(name, module):
    def hook_function(module, inputs, outputs):
        print('================================================')
        print(module)
        print_func(inputs, name + ' inputs')
        print_func(outputs, name + ' outputs')
    return hook_function

def main():
    global best_acc
    start_epoch = args.start_epoch  # start from epoch 0 or last checkpoint epoch

    if not os.path.isdir(args.checkpoint) and args.local_rank == 0:
        mkdir_p(args.checkpoint)

    args.distributed = True

    #args.local_rank = int(os.environ['LOCAL_RANK'])
    #args.world_size = int(os.environ['WORLD_SIZE']) 
    args.device_id = args.local_rank
    print("vvvvvvvvvvvvvvvvvv",args.device_id)
    args.device = '{}:{}'.format(args.device, args.local_rank)
    print("xxxxxx",args.device)
    torch.sdaa.set_device(args.device) if 'sdaa' in args.device else torch.cuda.set_device(args.device)
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '50001'
    dist.init_process_group(backend='tccl' if 'sdaa' in args.device else 'nccl', #init_method='env://',
                            rank=args.local_rank,world_size=args.world_size)


    # create model
    print("[{} #{}] => creating model".format(args.device,args.local_rank))

    model = pnasnet. pnasnet5large(1000, pretrained="", use_aux=args.use_aux)
    model.drop_path_prob = 0.4


    criterion = crossentropy.LabelSmoothingCrossEntropy(num_classes=1000).to(args.device)
    model = model.to(args.device)


    state['lr'] = args.lr

    optimizer = set_optimizer(model)

    scaler = amp.GradScaler(enabled=(args.opt_level != "O0"))
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], broadcast_buffers=False)


    # Data loading code
    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'val')


    data_aug_scale = (0.08, 1.0) 

    train_dataset = datasets.ImageFolder(traindir, transforms.Compose([
            transforms.RandomResizedCrop(331, scale=data_aug_scale),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            # normalize,
        ]))
    val_dataset = datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(int(331/0.875)),
            transforms.CenterCrop(331),
            transforms.ToTensor(),
            # normalize,
        ]))

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.train_batch,
        num_workers=args.workers, pin_memory=False, sampler=train_sampler,
        #collate_fn=fast_collate,
        drop_last=True)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.test_batch,
        num_workers=args.workers, pin_memory=False, sampler=val_sampler,
        #collate_fn=fast_collate,
        drop_last=True)


    # Resume
    title = 'ImageNet-pnasnet5large'
    if args.resume:
        print('==> Resuming from checkpoint..', args.resume)
        assert os.path.isfile(args.resume), 'Error: no checkpoint directory found!'
        print("=> loading checkpoint '{}'".format(args.resume))
        loc='{}'.format(args.device)
        checkpoint = torch.load(args.resume, map_location=loc)
        args.start_epoch = checkpoint['epoch']
        best_acc = checkpoint['best_acc']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])

        if 'scaler' in checkpoint:
            scaler.load_state_dict(checkpoint['scaler'])
            print("=> loaded scaler state from checkpoint")
        else:
            print("=> no scaler state found in checkpoint, starting from scratch")

        print("=> loaded checkpoint '{}' (epoch {})"
              .format(args.resume, checkpoint['epoch']))
    if args.local_rank == 0:
        logger = Logger(os.path.join(args.checkpoint, args.log), title=title)
        logger.set_names(['Learning Rate', 'Train Loss', 'Valid Loss', 'Train Acc.', 'Valid Acc.', 'Training FPS'])


    if args.evaluate:
        print('\nEvaluation only')
        test_loss, test_acc = test(val_loader, model, criterion, start_epoch, use_device)
        print(' Test Loss:  %.8f, Test Acc:  %.2f' % (test_loss, test_acc))
        if args.local_rank == 0:
            logger = Logger(os.path.join(args.checkpoint, args.log), title=title)
            logger.set_names(['Test Loss', 'Test Acc'])
            logger.append([test_loss, test_acc])
            logger.close()
        return



    # Train and val
    for epoch in range(start_epoch, args.epochs):
        train_sampler.set_epoch(epoch)

        adjust_learning_rate(optimizer, epoch, args)

        if args.local_rank == 0:
            print('\nEpoch: [%d | %d] LR: %f' % (epoch + 1, args.epochs, adjusted_lr))

        train_loss, train_acc, fps = train(train_loader, model, criterion, optimizer, epoch, use_device, args.world_size,scaler)
        test_loss, test_acc = test(val_loader, model, criterion, epoch, use_device)

        # save model
        if args.local_rank == 0:
            # append logger file
            logger.append([adjusted_lr, train_loss, test_loss, train_acc, test_acc, fps])

            is_best = test_acc > best_acc
            best_acc = max(test_acc, best_acc)

            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'acc': test_acc,
                'best_acc': best_acc,
                'optimizer': optimizer.state_dict(),
                'scaler': scaler.state_dict(),  # 新增：保存GradScaler状态
            }, is_best, checkpoint=args.checkpoint)

    if args.local_rank == 0:
        logger.close()

    print('Best acc:')
    print(best_acc)

def train(train_loader, model, criterion, optimizer, epoch, use_device, nnpus, scaler):
    printflag = False
    # switch to train mode
    model.train()
    torch.set_grad_enabled(True)

    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    fps = AverageMeter()
    ffps = AverageMeter()
    end = time.time()

    if args.local_rank == 0:
        bar = Bar('Processing', max=len(train_loader))
    show_step = len(train_loader) // 10
    
    # prefetcher = data_prefetcher(train_loader)
    # inputs, targets = prefetcher.next()

    #batch_idx = -1
    #while inputs is not None:
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        if args.max_step != 0 and batch_idx > args.max_step:
            exit()
        #batch_idx += 1
        batch_size = inputs.size(0)
        if batch_size < args.train_batch:
            break
        # measure data loading time

        print("Device string:", args.device)
        #loc = '{}:{}'.format(args.device,args.device_id)
        loc = '{}'.format(args.device)
        targets = targets.to(torch.int32)
        inputs, targets = inputs.to(loc, non_blocking=False), targets.to(loc, non_blocking=False)
        if (batch_idx) % show_step == 0 and args.local_rank == 0:
            print_flag = True
        else:
            print_flag = False
        with amp.autocast(enabled=(args.opt_level != "O0")):
            if args.cutmix:
                if printflag==False:
                    print('using cutmix !')
                    printflag=True
                inputs, targets_a, targets_b, lam = cutmix_data(inputs, targets, args.cutmix_prob, use_device)
                outputs = model(inputs)
                loss_func = mixup_criterion(targets_a, targets_b, lam)
                old_loss = loss_func(criterion, outputs)
            elif args.mixup:
                if printflag==False:
                    print('using mixup !')
                    printflag=True
                inputs, targets_a, targets_b, lam = mixup_data(inputs, targets, args.alpha, use_device)
                outputs = model(inputs)
                loss_func = mixup_criterion(targets_a, targets_b, lam)
                old_loss = loss_func(criterion, outputs)
            elif args.cutout:
                if printflag==False:
                    print('using cutout !')
                    printflag=True
                inputs = cutout_data(inputs, args.cutout_size, use_device)
                outputs = model(inputs)
                old_loss = criterion(outputs, targets)
            else:
                outputs, aux1 = model(inputs)
                if args.use_aux:
                    loss1 = criterion(outputs, targets)
                    loss2 = criterion(aux1, targets)
                    old_loss = loss1 + 0.4 * loss2
                else:
                    old_loss = criterion(outputs, targets)

        # compute gradient and do SGD step
        loss =old_loss
        optimizer.zero_grad()
        # loss.backward()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer) 
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10)
        if args.el2:
            optimizer.print_flag = print_flag

        scaler.step(optimizer) 
        scaler.update()

        if batch_idx % args.print_freq == 0:
            # measure accuracy and record loss
            prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
            reduced_loss = reduce_tensor(loss.data)
            prec1        = reduce_tensor(prec1)
            prec5        = reduce_tensor(prec5)

            # to_python_float incurs a host<->device sync
            losses.update(to_python_float(reduced_loss), inputs.size(0))
            top1.update(to_python_float(prec1), inputs.size(0))
            top5.update(to_python_float(prec5), inputs.size(0))

            if 'cuda' in args.device:
                torch.cuda.synchronize()
            if 'sdaa' in args.device:
                torch.sdaa.synchronize() 
            # measure elapsed time
            battime = (time.time() - end) / args.print_freq
            batch_time.update(battime)

            fps.update(batch_size / battime * nnpus)
            if batch_idx>10:
                ffps.update(batch_size / battime * nnpus)
            end = time.time()

            if args.local_rank == 0: # plot progress
                print( '({batch}/{size}) | Batch: {bt:.3f}s | Total: {total:} | OldLoss: {oldloss: .4f} |Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f} | fps: {fp: .1f} | rank:{rank} '.format(
                            batch=batch_idx + 1,
                            size=len(train_loader),
                            bt=batch_time.val,
                            total=bar.elapsed_td,
                            oldloss=old_loss,
                            loss=losses.avg,
                            top1=top1.avg,
                            top5=top5.avg,
                            fp=fps.val,
                            rank=args.local_rank
                            ))
                json_logger.log(
                    step = (epoch, batch_idx+1),
                data = {
                    "rank":args.local_rank,
                    "train.loss":losses.avg, 
                    "train.ips":fps.val,
                    },
                verbosity=Verbosity.DEFAULT,)                
                #bar.next()
        if (batch_idx) % show_step == 0 and args.local_rank == 0:
            print('E%d' % (epoch) + bar.suffix)

        #inputs, targets = prefetcher.next()

    return (losses.avg, top1.avg, ffps.avg)

def test(val_loader, model, criterion, epoch, use_device):
    global best_acc

    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()
    # torch.set_grad_enabled(False)

    end = time.time()
    if args.local_rank == 0:
        bar = Bar('Processing', max=len(val_loader))

    # prefetcher = data_prefetcher(val_loader)
    # inputs, targets = prefetcher.next()

    # batch_idx = -1
    # while inputs is not None:
    for batch_idx, (inputs, targets) in enumerate(val_loader):
        # batch_idx += 1

        loc = '{}'.format(args.device)
        targets = targets.to(torch.int32)
        inputs, targets = inputs.to(loc, non_blocking=False), targets.to(loc, non_blocking=False)

        # compute output
        with torch.no_grad():
            outputs, aux1 = model(inputs)
            loss = criterion(outputs, targets)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))

        reduced_loss = reduce_tensor(loss.data)
        prec1        = reduce_tensor(prec1)
        prec5        = reduce_tensor(prec5)

        # to_python_float incurs a host<->device sync
        losses.update(to_python_float(reduced_loss), inputs.size(0))
        top1.update(to_python_float(prec1), inputs.size(0))
        top5.update(to_python_float(prec5), inputs.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
        if args.local_rank == 0:
            print( 'Valid({batch}/{size}) | Batch: {bt:.3f}s | Total: {total:} | Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f} | rank:{rank} '.format(
                        batch=batch_idx + 1,
                        size=len(val_loader),
                        bt=batch_time.avg,
                        total=bar.elapsed_td,
                        loss=losses.avg,
                        top1=top1.avg,
                        top5=top5.avg,
                        rank=args.local_rank
                        ))

        # inputs, targets = prefetcher.next()

    return (losses.avg, top1.avg)

def save_checkpoint(state, is_best, checkpoint='checkpoint', filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'model_best.pth.tar'))


def set_optimizer(model):

    #optim_use = optimizers.NpuFusedSGD
    optim_use = optim.SGD
    if args.el2:
        optim_use = LSGD
        if args.local_rank == 0:
            print('use e-shifted L2 regularizer based SGD optimizer!')
    else:
        if args.local_rank == 0:
            print('use SGD optimizer!')


    if args.wdall:
        optimizer = optim_use(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
        print('weight decay on all parameters')
    else:
        decay_list = []
        no_decay_list = []
        dns = []
        ndns = []

        for name, p in model.named_parameters():
            no_decay_flag = False
            dim = p.dim()

            if 'bias' in name:
                no_decay_flag = True
            elif dim == 1:
                if args.nowd_bn: # bn weights
                    no_decay_flag = True
            elif dim == 2:
                if args.nowd_fc:  # fc weights
                    no_decay_flag = True
            elif dim == 4:
                if args.nowd_conv: # conv weights
                    no_decay_flag = True
            else:
                print('no valid dim!!!, dim = ', dim)
                exit(-1)

            if no_decay_flag:
                no_decay_list.append(p)
                ndns.append(name)
            else:
                decay_list.append(p)
                dns.append(name)

        if args.local_rank == 0:
            print('------------' * 6)
            print('no decay list = ', ndns)
            print('------------' * 6)
            print('decay list = ', dns)
            print('------summary------')
            if args.nowd_bn:
                print('no decay on bn weights!')
            else:
                print('decay on bn weights!')
            if args.nowd_conv:
                print('no decay on conv weights!')
            else:
                print('decay on conv weights!')
            if args.nowd_fc:
                print('no decay on fc weights!')
            else:
                print('decay on fc weights!')
            print('------------' * 6)

        params = [{'params': no_decay_list, 'weight_decay': 0},
                  {'params': decay_list}]
        optimizer = optim_use(params, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
        if args.local_rank == 0:
            print('optimizer = ', optimizer)

    return optimizer


def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    # lr = args.lr * (0.1 ** (epoch // (args.epochs//3 - 3)))

    if args.warmup > 0 and epoch < args.warmup:
        lr = args.lr * ((epoch + 1) / (args.warmup + 1))
    else:
        alpha = 0
        cosine_decay = 0.5 * (
                1 + np.cos(np.pi * (epoch - args.warmup) / (args.epochs - args.warmup)))
        decayed = (1 - alpha) * cosine_decay + alpha
        lr = args.lr * decayed

    print("=> Epoch[%d] Setting lr: %.4f" % (epoch, lr))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    global adjusted_lr
    adjusted_lr = lr

# class SoftCrossEntropyLoss(nn.NLLLoss):
#     def __init__(self, label_smoothing=0, num_classes=1000, **kwargs):
#         assert label_smoothing >= 0 and label_smoothing <= 1
#         super(SoftCrossEntropyLoss, self).__init__(**kwargs)
#         self.confidence = 1 - label_smoothing
#         self.other      = label_smoothing * 1.0 / (num_classes - 1)
#         self.criterion  = nn.KLDivLoss(reduction='batchmean')
#         print('using soft celoss!!!, label_smoothing = ', label_smoothing)
#
#     def forward(self, input, target):
#         one_hot = torch.zeros_like(input)
#         one_hot.fill_(self.other)
#         one_hot.scatter_(1, target.unsqueeze(1).long(), self.confidence)
#         input   = F.log_softmax(input, 1)
#         return self.criterion(input, one_hot)

def mixup_data(x, y, alpha=1.0, use_device=True):
    if alpha > 0.:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.

    batch_size = x.size(0)
    if use_device:
        index = torch.randperm(batch_size).npu()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, ...]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2

def cutmix_data(x, y, cutmix_prob=1.0, use_device=True):
    lam = np.random.beta(1, 1)

    batch_size = x.size(0)
    index = torch.randperm(batch_size).npu()
    y_a, y_b = y, y[index]

    bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
    x[:, :, bbx1:bbx2, bby1:bby2] = x[index, :, bbx1:bbx2, bby1:bby2]
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size()[-1] * x.size()[-2]))
    return x, y_a, y_b, lam

def cutout_data(x, cutout_size=112, use_device=True):
    W = x.size(2)
    H = x.size(3)
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cutout_size // 2, 0, W)
    bby1 = np.clip(cy - cutout_size // 2, 0, H)
    bbx2 = np.clip(cx + cutout_size // 2, 0, W)
    bby2 = np.clip(cy + cutout_size // 2, 0, H)

    x[:, :, bbx1:bbx2, bby1:bby2] = 0

    return x

def mixup_criterion(y_a, y_b, lam):
    return lambda criterion, pred: lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

def reduce_tensor(tensor):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= args.world_size
    return rt

if __name__ == '__main__':
    main()
