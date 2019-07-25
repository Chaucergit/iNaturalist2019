#!/usr/bin/python3
#coding=utf-8

import os
import sys
sys.path.insert(0, '../')
sys.dont_write_bytecode = True

import cv2
import pickle
import numpy as np
import matplotlib.pyplot as plt
plt.ion()

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import dataset
import network

def crop(image, mask, area):
    H, W  = mask.shape
    S     = int(np.sqrt(mask.sum()*area))
    X, Y  = np.where(mask==1)
    if len(X)<10 or len(Y)<10 or S<10:
        return F.interpolate(image, size=(448,448), mode='bilinear')
    randx = np.random.randint(len(X)//3, len(X)//3*2)
    randy = np.random.randint(len(Y)//3, len(Y)//3*2)
    xmin  = max(int(X[randx]-S/2), 0)
    ymin  = max(int(Y[randy]-S/2), 0)
    xmax  = min(int(X[randx]+S/2), H)
    ymax  = min(int(Y[randy]+S/2), W)
    return F.interpolate(image[:, :, xmin:xmax,ymin:ymax], size=(448,448), mode='bilinear')


def val(Dataset, Network):
    ## dataset
    cfg    = Dataset.Config(snapshot='./out/model-32', mode='val', batch=1)
    data   = Dataset.Data(cfg)
    loader = DataLoader(data, batch_size=cfg.batch, shuffle=False, num_workers=8, pin_memory=True)
    ## network
    net    = Network.Classify(cfg)
    net.train(False)
    net.cuda()
    
    with torch.no_grad():
        top1, top2, top3, top4, top5, total = 0, 0, 0, 0, 0, 0
        for step, (image, label, mask) in enumerate(loader):
            image, label, mask  = image.cuda().float(), label.cuda().long(), mask[0].numpy()
            rimage, rmask       = image.flip(-1), mask[:,::-1]

            tmp = torch.cat([ crop(image,  mask,  0.50),
                              crop(image,  mask,  0.50),
                              crop(image,  mask,  1.00),
                              crop(image,  mask,  1.00),
                              crop(image,  mask,  2.00),
                              crop(image,  mask,  2.00),
                              crop(image,  mask,  4.00),
                              crop(image,  mask,  4.00),
                              crop(rimage, rmask, 0.25),
                              crop(rimage, rmask, 0.25),
                              crop(rimage, rmask, 1.00),
                              crop(rimage, rmask, 1.00),
                              crop(rimage, rmask, 2.00),
                              crop(rimage, rmask, 2.00),
                              crop(rimage, rmask, 4.00),
                              crop(rimage, rmask, 4.00)], dim=0)
            out = F.softmax(net(tmp), dim=-1).mean(dim=0, keepdim=True)

            _, idx            = torch.sort(out/cfg.rate, dim=1, descending=True)
            top1             += (idx[:,:1] == label.unsqueeze(1)).sum().item()
            top2             += (idx[:,:2] == label.unsqueeze(1)).sum().item()
            top3             += (idx[:,:3] == label.unsqueeze(1)).sum().item()
            top4             += (idx[:,:4] == label.unsqueeze(1)).sum().item()
            top5             += (idx[:,:5] == label.unsqueeze(1)).sum().item()
            total            += out.size(0)
            print('step=%d | top1=%.6f | top2=%.6f | top3=%.6f | top4=%.6f | top5=%.6f'%(step*cfg.batch, top1/total, top2/total, top3/total, top4/total, top5/total))


if __name__=='__main__':
    val(dataset, network)
