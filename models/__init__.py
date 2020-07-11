
import torch

from .resnet import Res_net
from .resnet_att import Resnet_self_attention
from .pcb import pcb
from .pcb_rga import pcb_rga

__model_factory = {
    'Res_net': Res_net,
    'Resnet_self_attention': Resnet_self_attention,
    'pcb': pcb,
    'pcb_rga': pcb_rga
}


def build_model(name, num_classes, **kwargs):
    avai_models = list(__model_factory.keys())
    if name not in avai_models:
        raise KeyError(
            'Unknown model: {}. Must be one of {}'.format(name, avai_models)
        )
    return __model_factory[name](num_classes=num_classes, **kwargs)
