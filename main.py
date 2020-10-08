
import argparse
import os
import torch
import torch.nn as nn
from dataloader import getDataLoader
import dataloader
from models import build_model
from train_2output import train
from torch.backends import cudnn
from loss.crossEntropyLabelSmoothLoss import CrossEntropyLabelSmoothLoss
from loss.TripleLoss import TripletLoss

parser = argparse.ArgumentParser(description='Person ReID Frame')

# System parameters#System parameters-------------------------------------------------------------
parser.add_argument('--nThread', type=int, default=4, help='number of threads for data loading')
parser.add_argument('--nGPU', type=int, default=1, help='number of GPUs')
parser.add_argument('--save_path', type=str, default='../experiments')
parser.add_argument('--experiment', type=str, default='resnet50_cbam_reid_model_v4')

args = parser.parse_args()


if __name__ == "__main__":
    # devie-------------------------------------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Fix random seed---------------------------------------------------------------------------
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)

    # speed up compution---------------------------------------------------------------------------
    cudnn.benchmark = True

    # data------------------------------------------------------------------------------------
    train_loader, query_loader, gallery_loader, num_classes = getDataLoader(args)
    dataloader = [train_loader, query_loader, gallery_loader]

    # model------------------------------------------------------------------------------------
    model = build_model(num_classes=num_classes, args=args)
    model = model.to(device)

    # criterion-----------------------------------------------------------------------------------
    ce_labelsmooth_loss = CrossEntropyLabelSmoothLoss(num_classes=num_classes)
    MARGIN = 0.3
    triplet_loss = TripletLoss(margin=MARGIN)
    criterion = [ce_labelsmooth_loss, triplet_loss]

    # optimizer-----------------------------------------------------------------------------------
    base_param_ids = set(map(id, model.backbone.parameters()))
    new_params = [p for p in model.parameters() if id(p) not in base_param_ids]
    param_groups = [{'params': model.backbone.parameters(), 'lr': args.lr/10},
                    {'params': new_params, 'lr': args.lr}]
    optimizer = torch.optim.SGD(param_groups, momentum=0.9, weight_decay=5e-4, nesterov=True)

    # scheduler-----------------------------------------------------------------------------------
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

    # save_dir_path-----------------------------------------------------------------------------------
    save_dir_path = os.path.join(args.save_path, args.dataset_name)
    os.makedirs(save_dir_path, exist_ok=True)

    # train -----------------------------------------------------------------------------------
    train(model, criterion, optimizer, scheduler, dataloader, device, save_dir_path, args)
