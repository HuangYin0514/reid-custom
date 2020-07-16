from models import build_model
import torch

if __name__ == "__main__":
    model = build_model('pcb_rga_v5', num_classes=100, height=256, width=128)

    inp1 = torch.randn(3, 3, 256, 128)
    ##############################################
    out, out2 = model(inp1)
    # print([out_item.shape for out_item in out])
    print(out2.shape)  

    ##############################################
    # model.eval()
    # features = model(inp1)
    # print(features.shape)

    ##############################################
    print(model)
    # summary(model, (3, 256, 128))

    print('complete check.')
