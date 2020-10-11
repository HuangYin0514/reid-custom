# encoding: utf-8


import torchvision.transforms as T
from .transforms import RandomErasing


def build_transforms(args, is_train=True):
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]
    IMG_SIZE = (args.img_height, args.img_width)

    normalize_transform = T.Normalize(mean=MEAN, std=STD)

    if is_train:
        transform = T.Compose([
            T.Resize(IMG_SIZE, interpolation=3),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            normalize_transform,
            RandomErasing(probability=0.1, mean=MEAN)
        ])
    else:
        transform = T.Compose([
            T.Resize(IMG_SIZE, interpolation=3),
            T.ToTensor(),
            normalize_transform
        ])

    return transform
