import torch
from torch import nn
from torch.nn import functional as F
from torchvision import models
from utils import torchtool
import torch.utils.model_zoo as model_zoo


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class PCBModel(nn.Module):
    def __init__(self,
                 num_classes,
                 block,
                 layers,
                 parts=6,
                 reduced_dim=256,
                 nonlinear='relu',
                 ** kwargs):

        super(PCBModel, self).__init__()
        # setting of parameters--------------------------------------------------------------------------
        self.parts = parts

        # backbone network--------------------------------------------------------------------------
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, 64, layers[0])
        self.layer2 = self._make_layer(block, 256, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 512, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 1024, 512, layers[3], stride=1)

        self.backbone = nn.Sequential(
            self.conv1, self.bn1, self.relu, self.maxpool,
            self.layer1, self.layer2, self.layer3, self.layer4)

        ####################################################################################

        # avgpool--------------------------------------------------------------------------
        self.avgpool = nn.AdaptiveAvgPool2d((self.parts, 1))
        # self.dropout = nn.Dropout(p=0.5)

        # local_conv--------------------------------------------------------------------
        self.local_conv_list = nn.ModuleList()
        for _ in range(parts):
            local_conv = nn.Sequential(
                nn.Conv1d(2048, 256, kernel_size=1),
                nn.BatchNorm1d(256),
                nn.ReLU(inplace=True))
            local_conv.apply(torchtool.weights_init_kaiming)
            self.local_conv_list.append(local_conv)

        # Classifier for each stripe--------------------------------------------------------------------------
        self.fc_list = nn.ModuleList()
        for _ in range(parts):
            fc = nn.Linear(256, num_classes)
            fc.apply(torchtool.weights_init_classifier)
            self.fc_list.append(fc)

    # --------------------------------------------------------------------------
    def _make_layer(self, block, inplanes, planes, num_layers, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = []
        layers.append(block(inplanes, planes, stride, downsample))
        for i in range(1, num_layers):
            layers.append(block(planes * block.expansion, planes))
        return nn.Sequential(*layers)

    # --------------------------------------------------------------------------
    def featuremaps(self, x):
        x = self.backbone(x)
        return x

    # --------------------------------------------------------------------------
    def forward(self, x):
        # backbone------------------------------------------------------------------------------------
        # tensor T
        resnet_features = self.featuremaps(x)

        # tensor g---------------------------------------------------------------------------------
        # [N, C, H, W]
        features_G = self.avgpool(resnet_features)
        # features_G = self.dropout(features_G)

        # 1x1 conv---------------------------------------------------------------------------------
        # [N, C=256, H=S, W=1]
        features_H = []
        for i in range(self.parts):
            stripe_features_H = self.local_conv_list[i](features_G[:, :, i, :])
            features_H.append(stripe_features_H)

        # Return the features_H***********************************************************************
        if not self.training:
            v_g = torch.cat(features_H, dim=2)
            v_g = F.normalize(v_g, p=2, dim=1)
            return v_g.view(v_g.size(0), -1)

        # fc---------------------------------------------------------------------------------
        # [N, C=num_classes]
        batch_size = x.size(0)
        logits_list = [self.fc_list[i](features_H[i].view(batch_size, -1)) for i in range(self.parts)]

        return logits_list


def init_pretrained_weights(model, model_url):
    """Initializes model with pretrained weights.

    Layers that don't match with pretrained layers in name or size are kept unchanged.
    """
    pretrain_dict = model_zoo.load_url(model_url)
    model_dict = model.state_dict()
    pretrain_dict = {
        k: v
        for k, v in pretrain_dict.items()
        if k in model_dict and model_dict[k].size() == v.size()
    }
    # print(pretrain_dict.keys())
    model_dict.update(pretrain_dict)
    model.load_state_dict(model_dict)


def PCB_init(num_classes, num_stripes=6, pretrained=True, **kwargs):
    assert num_stripes == 6, "num_stripes not eq 6"

    model = PCBModel(
        num_classes=num_classes,
        block=Bottleneck,
        layers=[3, 4, 6, 3],
        parts=num_stripes,
        reduced_dim=256,
        nonlinear='relu',
        **kwargs
    )
    if pretrained:
        init_pretrained_weights(model, model_urls['resnet50'])

    return model
