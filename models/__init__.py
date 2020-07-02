
import torch

from .pcb import PCB_p6
from .resnet import Res_net
from .resnet_att import Resnet_self_attention
from .pcb_init import PCB_init
from .rga_branch import rga_branch
from .resnet50_rga_model import resnet50_rga_model


__model_factory = {
    'PCB_p6': PCB_p6,
    'Res_net': Res_net,
    'Resnet_self_attention': Resnet_self_attention,
    'PCB_init': PCB_init,
    'rga_branch': rga_branch,
    'resnet50_rga_model':resnet50_rga_model
}


def build_model(name, **kwargs):
    avai_models = list(__model_factory.keys())
    if name not in avai_models:
        raise KeyError(
            'Unknown model: {}. Must be one of {}'.format(name, avai_models)
        )
    return __model_factory[name](**kwargs)
