
import sys
sys.path.append('/home/hy/vscode/reid-custom')
print(sys.path)
from models import build_model
import torch

if __name__ == "__main__":

    model = build_model('PCB_p6', num_classes=64, share_conv=True)
    print(model)

    model.eval()
    inp1 = torch.randn(3,3,384,128)
    out = model(inp1)
    print([out_item.shape for out_item in out])


    print('complete check.')
