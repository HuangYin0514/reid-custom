import sys
sys.path.append('/home/hy/vscode/reid-custom')
print(sys.path)
import argparse
from dataloader import getDataLoader


parser = argparse.ArgumentParser(description='Person ReID Frame')

# Data parameters-------------------------------------------------------------
parser.add_argument('--dataset_name', type=str, default='market1501')
parser.add_argument('--dataset_path', type=str, default='/home/hy/vscode/data/Market-1501-v15.09.15')
parser.add_argument('--img_height', type=int, default=384, help='height of the input image')
parser.add_argument('--img_width', type=int, default=128, help='width of the input image')
parser.add_argument('--batch_size', default=6, type=int, help='batch_size')
parser.add_argument('--test_batch_size', default=6, type=int, help='test_batch_size')
parser.add_argument('--data_sampler_type', type=str, default='RandomIdentitySampler')
parser.add_argument('--num_instance', type=int, default=2)

# test other datset parameters-------------------------------------------------------------
parser.add_argument('--test_other_dataset_name', type=str, default='Occluded_REID')
parser.add_argument('--test_other_dataset_path', type=str, default='/home/hy/vscode/data/Occluded_REID')
args = parser.parse_args()


if __name__ == "__main__":
    # dataset
    train_loader, query_loader, gallery_loader = getDataLoader(args=args)
    for data in train_loader:
        img, pids = data
        print(img.shape)
    print(train_loader, query_loader, gallery_loader)
