import torch
from torch import nn
from torch.nn import functional as F
from torchvision import models
from utils import torchtool
import torch.utils.model_zoo as model_zoo
from .rga_branch import RGA_Branch

MODEL_URLS = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


# ===================
#   Initialization
# ===================
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
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)


# ===============
#    RGA Model
# ===============
class ResNet50_RGA_Model(nn.Module):
    '''
    Backbone: ResNet-50 + RGA modules.
    '''

    def __init__(self,
                 pretrained=True,
                 num_feat=2048,
                 height=256,
                 width=128,
                 dropout=0,
                 num_classes=0,
                 last_stride=1,
                 branch_name='rgas',
                 scale=8,
                 d_scale=8,
                 model_path=MODEL_URLS['resnet50'],
                 **kwargs):
        super(ResNet50_RGA_Model, self).__init__()
        self.pretrained = pretrained
        self.num_feat = num_feat
        self.dropout = dropout
        self.num_classes = num_classes
        self.branch_name = branch_name
        print('Num of features: {}.'.format(self.num_feat))

        # branch_name--------------------------------------------------------------------------
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

        # backbone--------------------------------------------------------------------------
        self.backbone = RGA_Branch(pretrained=pretrained, last_stride=last_stride,
                                   spa_on=spa_on, cha_on=cha_on, height=height, width=width,
                                   s_ratio=scale, c_ratio=scale, d_ratio=d_scale, model_path=model_path)

        ####################################################################################
        self.feat_bn = nn.BatchNorm1d(self.num_feat)
        self.feat_bn.bias.requires_grad_(False)
        if self.dropout > 0:
            self.drop = nn.Dropout(self.dropout)
        self.cls = nn.Linear(self.num_feat, self.num_classes, bias=False)

        self.feat_bn.apply(weights_init_kaiming)
        self.cls.apply(weights_init_classifier)

    def forward(self, inputs):
        # im_input = inputs[0]

        feat_ = self.backbone(inputs)
        
        feat_ = F.avg_pool2d(feat_, feat_.size()[2:]).view(feat_.size(0), -1)
        feat = self.feat_bn(feat_)
        if self.dropout > 0:
            feat = self.drop(feat)
        if self.training and self.num_classes is not None:
            cls_feat = self.cls(feat)
        
        if self.training:
            return (feat_, feat, cls_feat)
        else:
            return (feat_, feat)



def resnet50_rga_model(*args, **kwargs):
    model = ResNet50_RGA_Model(
        *args,
        **kwargs)
    return model
