from models import build_model
import torch
from torchsummary import summary

if __name__ == "__main__":

    model = build_model('pcb', height=256, width=128)

    # model.eval()
    # inp1 = torch.randn(3, 3, 256, 128)
    # out = model(inp1)

    # # print(model)
    # # print(out)
    # print([out_item.shape for out_item in out])
    # summary(model, (3, 256, 128))

    base_layers = [model.backbone.conv1, model.backbone.bn1,
                   model.backbone.layer1, model.backbone.layer2,
                   model.backbone.layer3, model.backbone.layer4]
    param_set = set()
    for i in base_layers:
        param_set = param_set | set(map(id, i.parameters()))
    print(param_set)

    print([1]+[2])

    print('complete check.')
