#!/usr/bin/python3
#coding=utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F

import math
import numpy as np
from collections import OrderedDict

class SEModule(nn.Module):
    def __init__(self, channels, reduction):
        super(SEModule, self).__init__()
        self.fc1      = nn.Conv2d(channels, channels//reduction, kernel_size=1, padding=0)
        self.fc2      = nn.Conv2d(channels//reduction, channels, kernel_size=1, padding=0)

    def forward(self, x):
        out = x.mean(dim=(2,3), keepdim=True)
        out = F.relu(self.fc1(out), inplace=True)
        out = torch.sigmoid(self.fc2(out))
        return x*out


class SEBottleneck(nn.Module):
    def __init__(self, inplanes, planes, groups, reduction, stride=1, downsample=None):
        super(SEBottleneck, self).__init__()
        self.conv1      = nn.Conv2d(inplanes, planes*2, kernel_size=1, bias=False)
        self.bn1        = nn.BatchNorm2d(planes*2)
        self.conv2      = nn.Conv2d(planes*2, planes*4, kernel_size=3, stride=stride, padding=1, groups=groups, bias=False)
        self.bn2        = nn.BatchNorm2d(planes*4)
        self.conv3      = nn.Conv2d(planes*4, planes*4, kernel_size=1, bias=False)
        self.bn3        = nn.BatchNorm2d(planes*4)
        self.relu       = nn.ReLU(inplace=True)
        self.se_module  = SEModule(planes*4, reduction=reduction)
        self.downsample = downsample
        self.stride     = stride

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1( x )), inplace=True)
        out = F.relu(self.bn2(self.conv2(out)), inplace=True)
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            x = self.downsample(x)
        out = F.relu(self.se_module(out)+x, inplace=True)
        return out


class SENet(nn.Module):
    def __init__(self, cfg):
        super(SENet, self).__init__()
        self.cfg      = cfg
        self.inplanes = 128
        self.layer0   = nn.Sequential(OrderedDict([
                        ('conv1', nn.Conv2d(3, 64, 3, stride=2, padding=1, bias=False)),
                        ('bn1'  , nn.BatchNorm2d(64)),
                        ('relu1', nn.ReLU(inplace=True)),
                        ('conv2', nn.Conv2d(64, 64, 3, stride=1, padding=1, bias=False)),
                        ('bn2'  , nn.BatchNorm2d(64)),
                        ('relu2', nn.ReLU(inplace=True)),
                        ('conv3', nn.Conv2d(64, self.inplanes, 3, stride=1, padding=1, bias=False)),
                        ('bn3'  , nn.BatchNorm2d(self.inplanes)),
                        ('relu3', nn.ReLU(inplace=True)),
                        ('pool' , nn.MaxPool2d(3, stride=2, ceil_mode=True))]))
        self.layer1   = self._make_layer(planes=64,  blocks=3,  stride=1, groups=64, reduction=16, downsample_kernel_size=1, downsample_padding=0)
        self.layer2   = self._make_layer(planes=128, blocks=8,  stride=2, groups=64, reduction=16, downsample_kernel_size=3, downsample_padding=1)
        self.layer3   = self._make_layer(planes=256, blocks=36, stride=2, groups=64, reduction=16, downsample_kernel_size=3, downsample_padding=1)
        self.layer4   = self._make_layer(planes=512, blocks=3,  stride=2, groups=64, reduction=16, downsample_kernel_size=3, downsample_padding=1)

    def _make_layer(self, planes, blocks, groups, reduction, stride, downsample_kernel_size, downsample_padding):
        downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes*4, kernel_size=downsample_kernel_size, stride=stride, padding=downsample_padding, bias=False), nn.BatchNorm2d(planes*4))
        layers     = [SEBottleneck(self.inplanes, planes, groups, reduction, stride, downsample)]
        self.inplanes = planes*4
        for _ in range(1, blocks):
            layers.append(SEBottleneck(self.inplanes, planes, groups, reduction))
        return nn.Sequential(*layers)

    def forward(self, x):
        out1 = self.layer0( x  )
        out2 = self.layer1(out1)
        out3 = self.layer2(out2)
        out4 = self.layer3(out3)
        out5 = self.layer4(out4)
        if self.cfg.mode == 'train':
            out5 = F.dropout2d(out5, p=0.25, training=True)
        return out1, out2, out3, out4, out5

    def initialize(self):
        self.load_state_dict(torch.load('../res/senet154-c7b49a05.pth'), strict=False)


class Classify(nn.Module):
    def __init__(self, cfg):
        super(Classify, self).__init__()
        self.cfg       = cfg
        self.bkbone    = SENet(cfg)
        self.linear    = nn.Conv2d(2048, cfg.cls_end-cfg.cls_beg, kernel_size=1, stride=1, padding=0)
        self.initialize()

    def forward(self, x):
        out1, out2, out3, out4, out5 = self.bkbone(x)
        return self.linear(out5)

    def initialize(self):
        if self.cfg.snapshot:
            self.load_state_dict(torch.load(self.cfg.snapshot))
        else:
            print('\nInitialize...')
            for n, m in self.named_children():
                print('initialize: '+n)
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
                elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.GroupNorm)):
                    nn.init.ones_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
                elif isinstance(m, nn.Linear):
                    nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
                else:
                    m.initialize()
