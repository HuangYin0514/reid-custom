
import torch

from .pcb import PCB_p6
from .resnet import Res_net
from .resnet_att import Resnet_self_attention


__model_factory = {
    'PCB_p6': PCB_p6,
    'Res_net': Res_net,
    'Resnet_self_attention': Resnet_self_attention
}


def build_model(name, num_classes, **kwargs):
    avai_models = list(__model_factory.keys())
    if name not in avai_models:
        raise KeyError(
            'Unknown model: {}. Must be one of {}'.format(name, avai_models)
        )
    return __model_factory[name](num_classes=num_classes, **kwargs)
