import sys
sys.path.append('/home/hy/vscode/reid-custom')
from utils import torchtool
from models import build_model


def check_layers_grad(model):
    for name, module in model.named_children():
        for p in module.parameters():
            print(name, p.requires_grad)
            break
    print('-'*10)   



if __name__ == "__main__":

    model = build_model('PCB_p6', num_classes=10, share_conv=False)
    print(model)

    open_layers = ['local_conv_list','fc_list']
    torchtool.open_specified_layers(model, open_layers)
    check_layers_grad(model)

    torchtool.open_all_layers(model)
    check_layers_grad(model)
    print('complete check.')
