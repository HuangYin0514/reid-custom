from torch import nn

__all__ = [
    'mylinearnnet_baisc'
]


# Convolutional neural network (two convolutional layers)
class mylinearnnet(nn.Module):
    def __init__(self, num_classes=10, **kwargs):
        super(mylinearnnet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc = nn.Linear(65536, num_classes)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        if not self.training:
            return out
        out = self.fc(out)
        return out


def mylinearnnet_baisc(num_classes, loss='softmax', pretrained=True, **kwargs):
    model = mylinearnnet(
        num_classes=num_classes,
        **kwargs
    )
    return model


if __name__ == "__main__":
    net = mylinearnnet_baisc(num_classes=9)
    print(net)
    import torch
    net.train()
    input = torch.randn(4, 3, 256, 128)
    print(net(input).shape)

    print('*'*100)

    net.eval()
    input = torch.randn(4, 3, 256, 128)
    print(net(input).shape)
