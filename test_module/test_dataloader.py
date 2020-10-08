import sys
sys.path.append('/home/hy/vscode/reid-custom')
print(sys.path)
import argparse
from dataloader import getDataLoader


if __name__ == "__main__":
    # dataset
    train_loader, query_loader, gallery_loader = getDataLoader(args=args)
    for data in train_loader:
        img, pids = data
        print(img.shape)
    print(train_loader, query_loader, gallery_loader)
