
from dataloader import getDataLoader


if __name__ == "__main__":
    # dataset
    dataloader = getDataLoader(
        'Occluded_REID', 4, '/home/hy/vscode/reid-custom/data/Occluded_REID', 'query', shuffle=True, augment=True)
    # dataloader = getDataLoader(
    #     'Market1501', 4, '/home/hy/vscode/reid-custom/data/Market-1501-v15.09.15', 'train', shuffle=True, augment=True)
    print()
    