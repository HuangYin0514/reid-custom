
import torch

from model import PCB_p6
from model import Res_net
from model import Resnet_self_attention


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



if __name__ == "__main__":
    model = build_model('Resnet_self_attention', num_classes=6)
    print(model)
    # test input and ouput
    input = torch.randn(4, 3, 384, 128)
    print(model(input))
    ## output is list
    if isinstance(model(input), list):
        print([k.shape for k in model(input)])
    else:
        print(model(input).shape)
