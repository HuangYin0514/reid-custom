from __future__ import division, absolute_import
from torch import nn
from torch.nn import functional as F


class PCBModel(nn.Module):
    def __init__(self,n ,**kwargs):
        super().__init__()
        self.print_kwarage(**kwargs)
        self.net = nn.Sequential(
            nn.Linear(128*384, 10),
            nn.ReLU(),
            nn.Linear(10, n),
            nn.ReLU(),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        return self.net(x)

    def print_kwarage(self,a=0,**kwargs):
        print(kwargs)


if __name__ == "__main__":
    # kwargs = {'a': 100, 'B': 'b', 'C': 'c', 'D': 'd'}
    # t = test(a=2,b=3)
    # import torch
    # input = torch.randn(5, 128*384)
    # print(t(input))
    t = PCBModel(
        6,
        layers=[2, 2, 2],
        channels=[64, 256, 384, 512],
    )
    import torch
    input = torch.randn(2, 128*384)
    print(t(input))
