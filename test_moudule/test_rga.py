import sys
sys.path.append('/home/hy/vscode/reid-custom/models')
print(sys.path)

import torch
from rga_module import RGA_Module



if __name__ == "__main__":
    # rga module--------------------------------------------------------------------------
    branch_name = 'rgac'
    if 'rgasc' in branch_name:
        spa_on = True
        cha_on = True
    elif 'rgas' in branch_name:
        spa_on = True
        cha_on = False
    elif 'rgac' in branch_name:
        spa_on = False
        cha_on = True
    else:
        raise NameError
    spa_on = False
    cha_on = True
    s_ratio = 6
    c_ratio = 6
    d_ratio = 6
    height = 384
    width = 128
    print((height//16)*(width//16))
    rga_att = RGA_Module(192, 24*6, use_spatial=spa_on, use_channel=cha_on,
                                cha_ratio=c_ratio, spa_ratio=s_ratio, down_ratio=d_ratio)

    input = torch.randn(6,512,24,6)
    output = rga_att(input)
    print(output.shape)
    print('complete check.')
