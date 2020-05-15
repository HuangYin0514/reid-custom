import os
import torch
from torchvision import datasets, transforms
from .occluded_reid import Occluded_REID
from .market1501 import Market1501

__dataset_factory = {
    'Occluded_REID': Occluded_REID,
    'Market1501':Market1501
}


# ---------------------- Global settings ----------------------
def getDataLoader(dataset, batch_size, dataset_path, part, shuffle=True, augment=True):
    # check ------------------------------------------------------------
    assert part in {'train', 'query', 'gallery'}, 'part not in folders'

    avai_dataset = list(__dataset_factory.keys())
    if dataset not in avai_dataset:
        raise KeyError(
            'Unknown model: {}. Must be one of {}'.format(name, avai_dataset)
        )
    # transform ------------------------------------------------------------
    transform_list = [
        transforms.Resize(size=(384, 128), interpolation=3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]

    if augment:
        transform_list.insert(1, transforms.RandomHorizontalFlip())

    data_transform = transforms.Compose(transform_list)
    # dataset ------------------------------------------------------------
    image_dataset = __dataset_factory[dataset](
        root=dataset_path, part=part, transform=data_transform)

    dataloader = torch.utils.data.DataLoader(image_dataset, batch_size=batch_size,
                                             shuffle=shuffle, num_workers=4)

    return dataloader


def check_data(images, fids, img_save_path):
    """
    check data of image
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import torchvision.utils as vutils

    # [weight, hight]
    plt.figure(figsize=(10, 10))
    plt.axis("off")
    plt.title(fids)
    plt.imshow(np.transpose(vutils.make_grid(images,
                                             padding=2,
                                             normalize=True),
                            (1, 2, 0)))
    plt.savefig(img_save_path)
