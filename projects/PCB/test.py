from __future__ import division, absolute_import
from torch import nn
from torch.nn import functional as F


class test(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        print(kwargs)
        self.print_kwarage(**kwargs)
        self.net = nn.Sequential(
            nn.Linear(128*384, 10),
            nn.ReLU(),
            nn.Linear(10, 1),
            nn.ReLU(),
            nn.Softmax()
        )

    def forward(self, x):
        return self.net(x)

    def print_kwarage(self,a=0,**kwargs):
        print(a)


if __name__ == "__main__":
    kwargs = {'a': 100, 'B': 'b', 'C': 'c', 'D': 'd'}
    t = test(a=100)
    import torch
    input = torch.randn(5, 128*384)
    print(t(input))
