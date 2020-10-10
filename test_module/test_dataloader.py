import sys
sys.path.append('/home/hy/vscode/reid-custom')
print(sys.path)
from dataloader import getDataLoader
import argparse

if __name__ == "__main__":
    # dataset
    train_loader, query_loader, gallery_loader, num_classes = getDataLoader(args.dataset_name, args.dataset_path, args=args)
    for data in train_loader:
        img, pids = data
        print(img.shape)
    print(train_loader, query_loader, gallery_loader)
