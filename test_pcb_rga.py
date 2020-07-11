from models import build_model
import torch

if __name__ == "__main__":
    model = build_model('pcb_rga', num_classes=100, height=384, width=128)

    model.eval()
    inp1 = torch.randn(3, 3, 384, 128)
    out = model(inp1)

    # # print(model)
    # # print(out)
    print([out_item.shape for out_item in out])
    # summary(model, (3, 256, 128))

    print('complete check.')
