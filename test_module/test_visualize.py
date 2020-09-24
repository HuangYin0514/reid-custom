import sys
sys.path.append('/home/hy/vscode/reid-custom')
print(sys.path)

import torch
from scheduler import build_scheduler
from models import build_model

from utils.visualize_ranked_results import visualize_ranked_results
from dataloader import getDataLoader
import argparse

# -----------------------------parameters setting --------------------------------
parser = argparse.ArgumentParser(description='Testing arguments')

parser.add_argument('--experiment', type=str, default='pcb_rga_v5')
parser.add_argument('--save_path', type=str, default='../experiments')
parser.add_argument('--which_epoch', default='final', type=str, help='0,1,2,3...or final')
parser.add_argument('--checkpoint', type=str, default='/home/hy/vscode/reid-custom/experiments/Market1501')

parser.add_argument('--dataset', type=str, default='Occluded_REID')
parser.add_argument('--dataset_path', type=str, default='/home/hy/vscode/data/Occluded_REID')
parser.add_argument('--height', type=int, default=384, help='height of the input image')
parser.add_argument('--width', type=int, default=128, help='width of the input image')

parser.add_argument('--batch_size', default=3, type=int, help='batchsize')
parser.add_argument('--share_conv', default=False, action='store_true')

args = parser.parse_args()

if __name__ == "__main__":

    model = build_model('resnet50_cbam_reid_model', num_classes=64)
    print(model)


    query_dataloader = getDataLoader(args.dataset, args.batch_size, args.dataset_path, 'query',   args, shuffle=False, augment=False)
    gallery_dataloader = getDataLoader(args.dataset, args.batch_size, args.dataset_path, 'gallery', args, shuffle=False, augment=False)