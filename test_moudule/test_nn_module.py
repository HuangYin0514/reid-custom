import torch
from torch import nn
from torch.nn import functional as F


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)
    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.normal_(m.weight, 1.0, 0.02)
            nn.init.constant_(m.bias, 0.0)


class Feature_Fusion_Module(nn.Module):
    # 自定义特征融合模块
    def __init__(self, ** kwargs):
        super(Feature_Fusion_Module, self).__init__()
        # Classifier for each stripe
        self.fc1 = nn.Linear(512, 6)
        self.fc1.apply(weights_init_kaiming)


    def forward(self, x):
        x = torch.sigmoid(self.fc1(x))
        return x


if __name__ == "__main__":
    ffm = Feature_Fusion_Module()
    print(ffm)
    rd = torch.randn(3, 512)
    print(rd.shape)
    res = ffm(rd)
    print(res)
    print(res.shape)
    print('complete check.')
