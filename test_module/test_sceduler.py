import sys
sys.path.append('/home/hy/vscode/reid-custom')
print(sys.path)

import torch
from scheduler import build_scheduler
from models import build_model



if __name__ == "__main__":

    model = build_model('PCB_p6', num_classes=64, share_conv=True)
    print(model)

    base_param_ids = set(map(id, model.backbone.parameters()))
    new_params = [p for p in model.parameters() if id(p) not in base_param_ids]
    param_groups = [{'params': model.backbone.parameters(), 'lr': 1000, 'lr_mult': 0.1},
                    {'params': new_params, 'lr': 1000, 'lr_mult': 1.0}]
    optimizer = torch.optim.SGD(param_groups, momentum=0.9, weight_decay=5e-4, nesterov=True)

    scheduler = build_scheduler('pcb_scheduler', optimizer=optimizer, lr=0.1)

    for g in optimizer.param_groups:
        print('old', g['lr'])

    for _ in range(60):
        scheduler.step(_)

    for g in optimizer.param_groups:
        print('new', g['lr'])

    print('complete check.')
