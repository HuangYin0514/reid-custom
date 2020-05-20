
from collections import OrderedDict
import shutil
import warnings
import os
import os.path as osp
from functools import partial
import pickle

import torch
import torch.nn as nn

#open_specified_layers-----------------------------------------------------------
def open_specified_layers(model, open_layers):
    r"""Opens specified layers in model for training while keeping
    other layers frozen.
    Args:
        model (nn.Module): neural net model.
        open_layers (str or list): layers open for training.
    Examples::
        >>> from torchreid.utils import open_specified_layers
        >>> # Only model.classifier will be updated.
        >>> open_layers = 'classifier'
        >>> open_specified_layers(model, open_layers)
        >>> # Only model.fc and model.classifier will be updated.
        >>> open_layers = ['fc', 'classifier']
        >>> open_specified_layers(model, open_layers)
    """
    if isinstance(model, nn.DataParallel):
        model = model.module

    if isinstance(open_layers, str):
        open_layers = [open_layers]

    for layer in open_layers:
        assert hasattr(model, layer), '"{}" is not an attribute of the model, please provide the correct name'.format(layer)

    for name, module in model.named_children():
        if name in open_layers:
            module.train()
            for p in module.parameters():
                p.requires_grad = True
        else:
            module.eval()
            for p in module.parameters():
                p.requires_grad = False


#open_all_layers-----------------------------------------------------------
def open_all_layers(model):
    r"""Opens all layers in model for training.
    Examples::
        >>> from torchreid.utils import open_all_layers
        >>> open_all_layers(model)
    """
    model.train()
    for p in model.parameters():
        p.requires_grad = True