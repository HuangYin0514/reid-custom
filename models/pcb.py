import torch
from torch import nn
from torch.nn import functional as F
from torchvision import models
from utils import torchtool


class PCBModel(nn.Module):
    def __init__(self, num_classes, num_stripes, share_conv, loss='softmax', ** kwargs):

        super(PCBModel, self).__init__()
        self.num_stripes = num_stripes
        self.num_classes = num_classes
        self.loss = loss

        # bone--------------------------------------------------------------------------
        resnet = models.resnet50(pretrained=True)
        # Modifiy the stride of last conv layer----------------------------
        resnet.layer4[0].downsample[0].stride = (1, 1)
        resnet.layer4[0].conv2.stride = (1, 1)
        # Remove avgpool and fc layer of resnet------------------------------
        # modules = list(resnet.children())[:-2]
        # self.backbone = nn.Sequential(*modules)
        self.backbone = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool,
            resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4)

        ####################################################################################

        # avgpool--------------------------------------------------------------------------
        self.avgpool = nn.AdaptiveAvgPool2d((self.num_stripes, 1))
        # self.dropout = nn.Dropout(p=0.5)

        # local_conv--------------------------------------------------------------------
        self.local_conv_list = nn.ModuleList()
        for _ in range(num_stripes):
            local_conv = nn.Sequential(
                nn.Conv2d(2048, 256, kernel_size=1, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True))
            local_conv.apply(torchtool.weights_init_kaiming)
            self.local_conv_list.append(local_conv)

        # Classifier for each stripe--------------------------------------------------------------------------
        self.fc_list = nn.ModuleList()
        for _ in range(num_stripes):
            fc = nn.Linear(256, num_classes)
            fc.apply(torchtool.weights_init_classifier)
            self.fc_list.append(fc)

    def forward(self, x):
        # backbone------------------------------------------------------------------------------------
        # tensor T
        resnet_features = self.backbone(x)

        # tensor g---------------------------------------------------------------------------------
        # [N, C, H, W]
        features_G = self.avgpool(resnet_features)
        # features_G = self.dropout(features_G)

        # 1x1 conv---------------------------------------------------------------------------------
        # [N, C=256, H=S, W=1]
        features_H = []
        for i in range(self.num_stripes):
            stripe_features_H = self.local_conv_list[i](features_G[:, :, i:i+1, :])
            features_H.append(stripe_features_H)

        # Return the features_H***********************************************************************
        if not self.training:
            v_g = torch.cat(features_H, dim=2)
            v_g = F.normalize(v_g, p=2, dim=1)
            return v_g.view(v_g.size(0), -1)

        # fc---------------------------------------------------------------------------------
        # [N, C=num_classes]
        batch_size = x.size(0)
        logits_list = [self.fc_list[i](features_H[i].view(batch_size, -1)) for i in range(self.num_stripes)]

        return logits_list


def PCB_p6(num_classes, num_stripes=6, **kwargs):
    assert num_stripes == 6, "num_stripes not eq 6"
    return PCBModel(
        num_classes=num_classes,
        num_stripes=num_stripes,
        **kwargs
    )
