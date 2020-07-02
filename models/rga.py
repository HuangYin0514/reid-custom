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


def weights_init_fc(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.normal_(m.weight, 1.0, 0.02)
            nn.init.constant_(m.bias, 0.0)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * 4)
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


class RGA(nn.Module):
    def __init__(self,
                 num_classes,
                 pretrained=True,
                 last_stride=1,
                 block=Bottleneck,
                 layers=[3, 4, 6, 3],
                 spa_on=True,
                 cha_on=True,
                 s_ratio=8,
                 c_ratio=8,
                 d_ratio=8,
                 height=256,
                 width=128,
                 model_path=model_urls['resnet50']):
        super(RGA, self).__init__()
        self.in_channels = 64

        # backbone network--------------------------------------------------------------------------
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=last_stride)

        self.backbone = nn.Sequential(
            self.conv1, self.bn1, self.relu, self.maxpool,
            self.layer1, self.layer2, self.layer3, self.layer4)

        # Load the pre-trained model weights----------------------------------------------------------
        if pretrained:
            self.load_specific_param(self.conv1.state_dict(), 'conv1', model_path)
            self.load_specific_param(self.bn1.state_dict(), 'bn1', model_path)
            self.load_partial_param(self.layer1.state_dict(), 1, model_path)
            self.load_partial_param(self.layer2.state_dict(), 2, model_path)
            self.load_partial_param(self.layer3.state_dict(), 3, model_path)
            self.load_partial_param(self.layer4.state_dict(), 4, model_path)

        ####################################################################################

        # avgpool--------------------------------------------------------------------------
        parts = 6
        self.avgpool = nn.AdaptiveAvgPool2d((parts, 1))
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

    def _make_layer(self, block, channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, channels * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(channels * block.expansion),
            )
        layers = []
        layers.append(block(self.in_channels, channels, stride, downsample))
        self.in_channels = channels * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.in_channels, channels))
        return nn.Sequential(*layers)

    def load_partial_param(self, state_dict, model_index, model_url):
        param_dict = model_zoo.load_url(model_url)
        for i in state_dict:
            key = 'layer{}.'.format(model_index)+i
            if key in param_dict:
                state_dict[i].copy_(param_dict[key])
        del param_dict

    def load_specific_param(self, state_dict, param_name, model_url):
        param_dict = model_zoo.load_url(model_url)
        for i in state_dict:
            key = param_name + '.' + i
            if key in param_dict:
                state_dict[i].copy_(param_dict[key])
        del param_dict

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # tensor g---------------------------------------------------------------------------------
        # [N, C, H, W]
        features_G = self.avgpool(x)
        # features_G = self.dropout(features_G)

        # 1x1 conv---------------------------------------------------------------------------------
        # [N, C=256, H=S, W=1]
        features_H = []
        parts = 6
        for i in range(parts):
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
        logits_list = [self.fc_list[i](features_H[i].view(batch_size, -1)) for i in range(parts)]

        return logits_list


def rga_init(num_classes, pretrained=True, **kwargs):

    model = RGA(
        num_classes=num_classes,
        pretrained=pretrained,
        **kwargs)

    return model
