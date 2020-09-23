
from .resnet import Res_net
from .resnet_att import Resnet_self_attention
from .pcb import pcb
from .pcb_rga import pcb_rga
from .pcb_rga_v2 import pcb_rga_v2
from .pcb_rga_v3 import pcb_rga_v3
from .pcb_rga_v4 import pcb_rga_v4
from .pcb_rga_v5 import pcb_rga_v5

__model_factory = {
    'Res_net': Res_net,
    'Resnet_self_attention': Resnet_self_attention,
    'pcb': pcb,
    'pcb_rga': pcb_rga,
    'pcb_rga_v2':pcb_rga_v2,
    'pcb_rga_v3':pcb_rga_v3,
    'pcb_rga_v4':pcb_rga_v4,
    'pcb_rga_v5':pcb_rga_v5
}


def build_model(name, num_classes, **kwargs):
    avai_models = list(__model_factory.keys())
    if name not in avai_models:
        raise KeyError(
            'Unknown model: {}. Must be one of {}'.format(name, avai_models)
        )
    return __model_factory[name](num_classes=num_classes, **kwargs)
