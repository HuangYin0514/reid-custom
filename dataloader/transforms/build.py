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
            T.Resize(IMG_SIZE),
            T.RandomHorizontalFlip(),
            # T.Pad(cfg.INPUT.PADDING),
            # T.RandomCrop(cfg.INPUT.SIZE_TRAIN),
            T.ToTensor(),
            normalize_transform,
            # RandomErasing(probability=cfg.INPUT.RE_PROB, mean=cfg.INPUT.PIXEL_MEAN)
        ])
    else:
        transform = T.Compose([
            T.Resize(IMG_SIZE),
            T.ToTensor(),
            normalize_transform
        ])

    return transform
