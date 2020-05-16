
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from dataloader import getDataLoader
from models import build_model
from train import *

parser = argparse.ArgumentParser(description='Person ReID Frame')

"""
System parameters
"""
parser.add_argument('--nThread', type=int, default=4, help='number of threads for data loading')
parser.add_argument('--nGPU', type=int, default=1, help='number of GPUs')
parser.add_argument('--save_path', type=str, default='./experiments')
parser.add_argument('--experiment', type=str, default='PCB_p6')


"""
Data parameters
"""
parser.add_argument('--dataset', type=str, default='Market1501')
parser.add_argument('--dataset_path', type=str, default='/home/hy/vscode/reid-custom/data/Market-1501-v15.09.15')
parser.add_argument('--height', type=int, default=384, help='height of the input image')
parser.add_argument('--width', type=int, default=128, help='width of the input image')
parser.add_argument('--batch_size', default=64, type=int, help='batch_size')

"""
Model parameters
"""
parser.add_argument('--share_conv', default=False, action='store_true')
parser.add_argument('--stripes', type=int, default=6)


"""
Train parameters
"""
parser.add_argument('--epochs', type=int, default=60)
parser.add_argument('--test_every', type=int, default=10)


"""
Optimizer parameters
"""
parser.add_argument('--lr', type=float, default=0.1)
parser.add_argument('--lr_base', type=float, default=0.01)


"""
Learning rate parameters
"""
parser.add_argument('--decay_every', type=int, default=20)
parser.add_argument('--gamma', type=float, default=0.1)

args = parser.parse_args()


if __name__ == "__main__":
    # devie---------------------------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Fix random seed---------------------------------------------------------------------------
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)

    

    # dataset------------------------------------------------------------------------------------
    train_dataloader = getDataLoader(args.dataset, args.batch_size, args.dataset_path, 'train', shuffle=True, augment=True)

    # model------------------------------------------------------------------------------------
    model = build_model(args.experiment, num_classes=train_dataloader.dataset.num_train_pids, share_conv=args.share_conv)
    model = model.to(device)

    # criterion-----------------------------------------------------------------------------------
    criterion = nn.CrossEntropyLoss()

    # optimizer-----------------------------------------------------------------------------------
    base_param_ids = set(map(id, model.backbone.parameters()))
    new_params = [p for p in model.parameters() if id(p) not in base_param_ids]
    param_groups = [{'params': model.backbone.parameters(), 'lr_mult': 0.1},
                    {'params': new_params, 'lr_mult': 1.0}]
    optimizer = torch.optim.SGD(param_groups, lr=args.lr, momentum=0.9, weight_decay=5e-4, nesterov=True)

    # scheduler-----------------------------------------------------------------------------------
    scheduler = None

    # train -----------------------------------------------------------------------------------
    trainer = Train()
    trainer.train(model, criterion, optimizer, scheduler, train_dataloader, args.epochs, device, args)
