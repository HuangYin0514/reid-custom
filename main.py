
import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from dataloader import getDataLoader
from models import build_model
from scheduler import build_scheduler, LRScheduler
from train import train
from loss import loss_set

parser = argparse.ArgumentParser(description='Person ReID Frame')


# System parameters#System parameters-------------------------------------------------------------
parser.add_argument('--nThread', type=int, default=4, help='number of threads for data loading')
parser.add_argument('--nGPU', type=int, default=1, help='number of GPUs')
parser.add_argument('--save_path', type=str, default='./experiments')
parser.add_argument('--experiment', type=str, default='resnet50_rga_model')
parser.add_argument('--seed', type=int, default=16)


# Data parameters-------------------------------------------------------------
parser.add_argument('--dataset', type=str, default='Market1501')
parser.add_argument('--dataset_path', type=str, default='/home/hy/vscode/reid-custom/data/Market-1501-v15.09.15')
parser.add_argument('--height', type=int, default=256, help='height of the input image')
parser.add_argument('--width', type=int, default=128, help='width of the input image')
parser.add_argument('--batch_size', default=64, type=int, help='batch_size')

# Model parameters-------------------------------------------------------------
parser.add_argument('--share_conv', default=False, action='store_true')
parser.add_argument('--stripes', type=int, default=6)
parser.add_argument('--open_layers', nargs='+', default=[])


# Train parameters-------------------------------------------------------------
parser.add_argument('--epochs', type=int, default=1)
parser.add_argument('--test_every', type=int, default=1)
parser.add_argument('--fixbase_epoch', type=int, default=0)


# Optimizer parameters-------------------------------------------------------------
parser.add_argument('--lr', type=float, default=0.1)


# Learning rate parameters-------------------------------------------------------------
parser.add_argument('--decay_every', type=int, default=20)
parser.add_argument('--gamma', type=float, default=0.1)

args = parser.parse_args()


if __name__ == "__main__":
    # devie-------------------------------------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Fix random seed---------------------------------------------------------------------------
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # dataset------------------------------------------------------------------------------------
    train_dataloader = getDataLoader(args.dataset, args.batch_size, args.dataset_path, 'train',  args)

    # model------------------------------------------------------------------------------------
    model = build_model(args.experiment, num_classes=train_dataloader.dataset.num_train_pids)
    model = model.to(device)

    # criterion-----------------------------------------------------------------------------------
    criterion_cls = loss_set.CrossEntropyLabelSmoothLoss(dataset.num_train_pids).cuda()
    criterion_tri = loss_set.TripletHardLoss(margin=0.3)
    criterion = [criterion_cls, criterion_tri]

    # optimizer-----------------------------------------------------------------------------------
    base_param_ids = set(map(id, model.backbone.parameters()))
    new_params = [p for p in model.parameters() if id(p) not in base_param_ids]
    param_groups = [{'params': model.backbone.parameters(), 'lr_mult': 1.0},
                    {'params': new_params, 'lr_mult': 1.0}]
    optimizer = torch.optim.SGD(param_groups,  lr=args.lr, momentum=0.9, weight_decay=5e-4, nesterov=True)

    # scheduler-----------------------------------------------------------------------------------
    lr_scheduler = LRScheduler(base_lr=0.0008, step=[80, 120, 160, 200, 240, 280, 320, 360],
                               factor=0.5, warmup_epoch=20,
                               warmup_begin_lr=0.000008)

    # save_dir_path-----------------------------------------------------------------------------------
    save_dir_path = os.path.join(args.save_path, args.dataset)
    os.makedirs(save_dir_path, exist_ok=True)

    # train -----------------------------------------------------------------------------------
    train(model, criterion, optimizer, scheduler, train_dataloader, args.epochs, device, save_dir_path, args)
