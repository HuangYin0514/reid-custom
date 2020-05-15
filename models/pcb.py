import torch
from torch import nn
from torch.nn import functional as F
from torchvision import models


class PCBModel(nn.Module):
    def __init__(self,
                 num_classes,
                 num_stripes,
                 share_conv,
                 loss='softmax',
                 ** kwargs):

        super(PCBModel, self).__init__()
        self.num_stripes = num_stripes
        self.num_classes = num_classes
        self.share_conv = share_conv
        self.loss = loss

        resnet = models.resnet50(pretrained=True)
        # Modifiy the stride of last conv layer
        resnet.layer4[0].conv2 = nn.Conv2d(
            512, 512, kernel_size=3, bias=False, stride=1, padding=1)
        resnet.layer4[0].downsample = nn.Sequential(
            nn.Conv2d(1024, 2048, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(2048))

        # Remove avgpool and fc layer of resnet
        modules = list(resnet.children())[:-2]
        self.backbone = nn.Sequential(*modules)

        # Add new layers
        self.avgpool = nn.AdaptiveAvgPool2d((self.num_stripes, 1))

        if share_conv:
            self.local_conv = nn.Sequential(
                nn.Conv2d(2048, 256, kernel_size=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True))
        else:
            self.local_conv_list = nn.ModuleList()
            for _ in range(num_stripes):
                local_conv = nn.Sequential(
                    nn.Conv1d(2048, 256, kernel_size=1),
                    nn.BatchNorm1d(256),
                    nn.ReLU(inplace=True))
                self.local_conv_list.append(local_conv)

        # Classifier for each stripe
        self.fc_list = nn.ModuleList()
        for _ in range(num_stripes):
            fc = nn.Linear(256, num_classes)

            nn.init.normal_(fc.weight, std=0.001)
            nn.init.constant_(fc.bias, 0)

            self.fc_list.append(fc)

    def forward(self, x):
        resnet_features = self.backbone(x)

        # [N, C, H, W]
        assert resnet_features.size(
            2) % self.num_stripes == 0, 'Image height cannot be divided by num_strides'

        features_G = self.avgpool(resnet_features)

        # [N, C=256, H=S, W=1]
        if self.share_conv:
            features_H = self.local_conv(features_G)
            features_H = [features_H[:, :, i, :]
                          for i in range(self.num_stripes)]
        else:
            features_H = []

            for i in range(self.num_stripes):
                stripe_features_H = self.local_conv_list[i](
                    features_G[:, :, i, :])
                features_H.append(stripe_features_H)

        # Return the features_H
        if not self.training:
            return torch.stack(features_H, dim=2)

        # [N, C=num_classes]
        batch_size = x.size(0)
        logits_list = [self.fc_list[i](features_H[i].view(batch_size, -1))
                       for i in range(self.num_stripes)]

        if self.loss == 'softmax':
            return logits_list
        elif self.loss == 'triplet':
            return logits_list, torch.stack(features_H, dim=2)
        else:
            raise KeyError("Unsupported loss: {}".format(self.loss))

        return logits_list




def PCB_p6(num_classes, share_conv, num_stripes=6, **kwargs):
    assert num_stripes == 6, "num_stripes not eq 6"
    return PCBModel(
        num_classes=num_classes,
        num_stripes=num_stripes,
        share_conv=share_conv,
        **kwargs
    )
