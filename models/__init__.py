from .pcb import pcb
from .rga_branch import rga_branch
from .resnet50_rga_model import resnet50_rga_model
from .pcb_rga_v2 import pcb_rga_v2

__model_factory = {
    'pcb': pcb,
    'rga_branch': rga_branch,
    'resnet50_rga_model': resnet50_rga_model,
    'pcb_rga_v2': pcb_rga_v2
}


def build_model(name, **kwargs):
    avai_models = list(__model_factory.keys())
    if name not in avai_models:
        raise KeyError(
            'Unknown model: {}. Must be one of {}'.format(name, avai_models)
        )
    return __model_factory[name](**kwargs)
