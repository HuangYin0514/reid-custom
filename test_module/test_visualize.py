import sys
sys.path.append('/home/hy/vscode/reid-custom')
print(sys.path)

import torch
from scheduler import build_scheduler
from models import build_model

from utils.visualize_ranked_results import visualize_ranked_results
from dataloader import getDataLoader
import argparse


if __name__ == "__main__":

    model = build_model('resnet50_cbam_reid_model', num_classes=64)
    print(model)


    query_dataloader = getDataLoader(args.dataset, args.batch_size, args.dataset_path, 'query',   args, shuffle=False, augment=False)
    gallery_dataloader = getDataLoader(args.dataset, args.batch_size, args.dataset_path, 'gallery', args, shuffle=False, augment=False)