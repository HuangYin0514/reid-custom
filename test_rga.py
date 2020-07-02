
from models import build_model
import torch
from torchsummary import summary

if __name__ == "__main__":

    model = build_model('rga_branch')

    model.eval()
    inp1 = torch.randn(3, 3, 384, 128)
    out = model(inp1)

    # print(model)
    # print([out_item.shape for out_item in out])
    summary(model, (3, 64, 64))

    print('complete check.')
