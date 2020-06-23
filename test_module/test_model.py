import sys
sys.path.append('/home/hy/vscode/reid-custom')
print(sys.path)
from torchsummary import summary
import torch
from models import build_model

if __name__ == "__main__":

    model = build_model('PCB_init', num_classes=64, share_conv=True)

    model.eval()
    inp1 = torch.randn(3, 3, 384, 128)
    out = model(inp1)

    # print(model)
    # print([out_item.shape for out_item in out])
    # summary(model, (3, 64, 64))

    print('complete check.')