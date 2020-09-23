import sys
sys.path.append('/home/hy/vscode/reid-custom')
print(sys.path)

import torch
from scheduler import build_scheduler
from models import build_model



if __name__ == "__main__":

    model = build_model('resnet50_cbam_reid_model', num_classes=64)
    print(model)

    print('*'*1000)
    # model = build_model('pcb', num_classes=64)
    # print(model)

    print('complete check.')
