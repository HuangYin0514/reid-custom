import sys
sys.path.append('/home/hy/vscode/reid-custom')
print(sys.path)
from dataloader import getDataLoader
import argparse


parser = argparse.ArgumentParser(description='Person ReID Frame')

# System parameters#System parameters-------------------------------------------------------------
parser.add_argument('--nThread', type=int, default=4, help='number of threads for data loading')
parser.add_argument('--nGPU', type=int, default=1, help='number of GPUs')
parser.add_argument('--save_path', type=str, default='../experiments')
parser.add_argument('--experiment', type=str, default='resnet50_cbam_reid_model_v4')

# Data parameters-------------------------------------------------------------
parser.add_argument('--dataset_name', type=str, default='market1501')
parser.add_argument('--dataset_path', type=str, default='/home/hy/vscode/data/Market-1501-v15.09.15')
parser.add_argument('--img_height', type=int, default=384, help='height of the input image')
parser.add_argument('--img_width', type=int, default=128, help='width of the input image')
parser.add_argument('--batch_size', default=6, type=int, help='batch_size')
parser.add_argument('--test_batch_size', default=6, type=int, help='test_batch_size')
parser.add_argument('--data_sampler_type', type=str, default='softmax')
parser.add_argument('--num_instance', type=int, default=2)


# Model parameters-------------------------------------------------------------
parser.add_argument('--stripes', type=int, default=6)


# Train parameters-------------------------------------------------------------
parser.add_argument('--epochs', type=int, default=2)
parser.add_argument('--test_every', type=int, default=2)
parser.add_argument('--fixbase_epoch', type=int, default=0)


# Optimizer parameters-------------------------------------------------------------
parser.add_argument('--lr', type=float, default=0.1)


# Learning rate parameters-------------------------------------------------------------
parser.add_argument('--decay_every', type=int, default=20)
parser.add_argument('--gamma', type=float, default=0.1)

# test other datset parameters-------------------------------------------------------------
parser.add_argument('--test_other_dataset_name', type=str, default='Occluded_REID')
parser.add_argument('--test_other_dataset_path', type=str, default='/home/hy/vscode/data/Occluded_REID')

args = parser.parse_args()

if __name__ == "__main__":
    # dataset
    train_loader, query_loader, gallery_loader, num_classes = getDataLoader(args.dataset_name, args.dataset_path, args=args)
    for data in train_loader:
        img, pids = data
        print(img.shape)
    print(train_loader, query_loader, gallery_loader)
