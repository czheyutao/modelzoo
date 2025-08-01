import os
import argparse
import torch
import torch.optim as optim
import warnings
# our module 
#from model_zoo import vgg16
#from model_zoo.pytorch_pretrained_vit import ViT
from model_zoo.swin.swin_transformer import get_swin
from model_zoo.pytorch_resnest.resnest.torch import resnest269
from base.trainer import BaseTrainer
from base.dataset import FoodDataset,ChunkSampler,P1_Dataset
from util import *

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    # training related argument
    parser.add_argument("-cont", "--cont",action="store_true", help='')
    parser.add_argument("-lr", "--lr", default=1e-6,type=float , help='')
    parser.add_argument("-period", "--period", default=20,type=int , help='')
    parser.add_argument("-batch_size", "--batch_size", default=8,type=int , help='')
    parser.add_argument("-gradaccum_size", "--gradaccum_size", default=1,type=int , help='')
    parser.add_argument("-load", "--load",default="",type=str , help='')
    parser.add_argument("-model_path", "--model_path",default="baseline",type=str , help='')
    parser.add_argument("-model_type", "--model_type",default="RESNEST269",type=str , help='')
    parser.add_argument("-max_epoch", "--max_epoch",default=100,type=int, help='')
    # data related argument
    parser.add_argument("-img_size", "--img_size", default=50,type=int , help='')
    parser.add_argument("-train_data_dir","--train_data_dir", default = "food_data/train",type=str, help ="Training images directory")
    parser.add_argument("-val_data_dir","--val_data_dir", default = "food_data/val",type=str, help ="Validation images directory")
    args = parser.parse_args()
    #######################
    # Environment setting
    #######################
    device = model_setting()
    fix_seeds(87)
    os.makedirs(args.model_path, exist_ok=True)
    ##############
    # Dataset
    ##############
    #train_dataset = P1_Dataset("hw1_data/train_50",val_mode=False)
    #val_dataset = P1_Dataset("hw1_data/val_50",val_mode=True)
    train_dataset = FoodDataset(args.train_data_dir,img_size=args.img_size,mode = "train")
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                                batch_size=args.batch_size,
                                                shuffle=True,
                                                num_workers=4)
                                                #sampler=ChunkSampler(1024, 512))
    val_dataset = FoodDataset(args.val_data_dir,img_size=args.img_size,mode = "val")
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                                batch_size=args.batch_size,
                                                shuffle=False,
                                                num_workers=4)
                                                #sampler=ChunkSampler(512, 0))
    ##############
    # Model
    ##############

    # TODO define ours' model,schedular
    # model = ViT(model_name, pretrained=True,num_classes=1000,image_size=384)
	# ResNeSt50
	# model = resnest50(pretrained=False)
	# model.load_state_dict(torch.load('./model_zoo/pytorch_resnest/resnest50_v1.pth'))
    # Swin Tranformer
    # model = get_swin(ckpt='./model_zoo/swin/swin_large_patch4_window12_384_22kto1k.pth')
    if args.model_type == "RESNEST269":
        model = resnest269(pretrained=True)
    elif args.model_type == "SWIN":
        model = get_swin(ckpt='./model_zoo/swin/swin_large_patch4_window12_384_22kto1k.pth')
    elif args.model_type == "SWIN_BBN":
        model = get_swin_bbn(ckpt='./model_zoo/swin/swin_large_patch4_window12_384_22kto1k.pth')
        model = BNNetwork(backbone_model=model,num_classes=1000,mode="swin") # Support swin/ResNet/ViT
    else:
        print("Wrong Model type QQ")
        assert(False)

    if args.load:
        model.load_state_dict(torch.load(args.load))
        print("model loaded from {}".format(args.load))

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr) #,weight_decay=0.01
    criterion =  torch.nn.CrossEntropyLoss()
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2)

    ##############
    # Trainer
    ##############
    trainer = BaseTrainer(
                 device = device, 
                 model = model,
                 optimizer = optimizer,
                 scheduler = None,
                 MAX_EPOCH = args.max_epoch,
                 criterion = criterion,
                 train_loader = train_loader,
                 val_loader = val_loader,
                 model_path = args.model_path,
                 lr = args.lr,
                 batch_size = args.batch_size, 
                 gradaccum_size = args.gradaccum_size, 
                 save_period = 10)
    trainer.train()

