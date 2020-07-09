from .pcb import pcb_p6
from .rga_branch import rga_branch
from .resnet50_rga_model import resnet50_rga_model
from .pcb_rga import pcb_rga

__model_factory = {
    'pcb_p6': pcb_p6,
    'rga_branch': rga_branch,
    'resnet50_rga_model': resnet50_rga_model,
    'pcb_rga': pcb_rga
}


def build_model(name, **kwargs):
    avai_models = list(__model_factory.keys())
    if name not in avai_models:
        raise KeyError(
            'Unknown model: {}. Must be one of {}'.format(name, avai_models)
        )
    return __model_factory[name](**kwargs)
