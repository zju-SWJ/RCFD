'''
VGG16 for CIFAR-10/100 Dataset.

Reference:
1. https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py

'''

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['vgg13', 'vgg16', 'vgg19']

#cfg = {
#    16: [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
#    19: [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
#}

class SepConv(nn.Module):
    def __init__(self, channel_in, channel_out, kernel_size=3, stride=2, padding=1, affine=True):
        super(SepConv, self).__init__()
        self.op = nn.Sequential(
            nn.Conv2d(channel_in, channel_in, kernel_size=kernel_size, stride=stride, padding=padding, groups=channel_in, bias=False),
            nn.Conv2d(channel_in, channel_in, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(channel_in, affine=affine),
            nn.ReLU(inplace=False),
            nn.Conv2d(channel_in, channel_in, kernel_size=kernel_size, stride=1, padding=padding, groups=channel_in, bias=False),
            nn.Conv2d(channel_in, channel_out, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(channel_out, affine=affine),
            nn.ReLU(inplace=False),
        )

    def forward(self, x):
        return self.op(x)

class VGG(nn.Module):
    def __init__(self, num_classes=100, depth=16, dropout = 0.0, is_big_size=False):
        super(VGG, self).__init__()
        self.inplances = 64
        self.conv1 = nn.Conv2d(3, self.inplances, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(self.inplances)
        self.conv2 = nn.Conv2d(self.inplances, self.inplances, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(self.inplances)
        self.relu = nn.ReLU(True)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.layer1 = self._make_layers(128, 2)
        if depth == 13:
            num_layer = 2
        elif depth == 16:
            num_layer = 3
        elif depth == 19:
            num_layer = 4
        self.layer2 = self._make_layers(256, num_layer)
        self.layer3 = self._make_layers(512, num_layer)
        self.layer4 = self._make_layers(512, num_layer)

        if is_big_size == False:
            self.fc = nn.Sequential(
                nn.Linear(512, 512),
                nn.ReLU(True),
                nn.Dropout(p = dropout),
                nn.Linear(512, 512),
                nn.ReLU(True),
                nn.Dropout(p = dropout),
                nn.Linear(512, num_classes)
            )
        else:
            self.fc = nn.Sequential(
                nn.Linear(7*7*512, 4096),
                nn.ReLU(True),
                nn.Dropout(p = dropout),
                nn.Linear(4096, 4096),
                nn.ReLU(True),
                nn.Dropout(p = dropout),
                nn.Linear(4096, num_classes)
            )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            #elif isinstance(m, nn.Linear):
            #    nn.init.normal_(m.weight, 0, 0.01)
            #    nn.init.constant_(m.bias, 0)
    
    def _make_layers(self, input, num_layer):    
        layers=[]
        for i in range(num_layer):
            conv2d = nn.Conv2d(self.inplances, input, kernel_size=3, padding=1)
            layers += [conv2d, nn.BatchNorm2d(input), nn.ReLU(inplace=True)]
            self.inplances = input
        layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        out_feature = x.view(x.size(0), -1)

        out = self.fc(out_feature)

        return out, out_feature

def vgg13(**kwargs):
    """
    Constructs a VGG13 model.
    """
    model = VGG(depth=13, **kwargs)
    return model
    
def vgg16(**kwargs):
    """
    Constructs a VGG16 model.
    """
    model = VGG(depth=16, **kwargs)
    return model
    
def vgg19(**kwargs):
    """
    Constructs a VGG19 model.
    """
    model = VGG(depth=19, **kwargs)
    return model