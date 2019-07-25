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


def test(Dataset, Network):
    ## dataset
    cfg    = Dataset.Config(snapshot='./out/model-32', mode='test', batch=1)
    data   = Dataset.Data(cfg)
    loader = DataLoader(data, batch_size=cfg.batch, shuffle=False, num_workers=16, pin_memory=True)
    ## network
    net    = Network.Classify(cfg)
    net.train(False)
    net.cuda()
    
    cnt = 0
    with torch.no_grad():
        top1, top2, top3, top4, top5, total = 0, 0, 0, 0, 0, 0
        with open('submission.csv', 'w') as f:
            f.write('id,predicted\n')
            for step, (image, imgID, mask) in enumerate(loader):
                image, mask   = image.cuda().float(), mask[0].numpy()
                rimage, rmask = image.flip(-1), mask[:,::-1]

                tmp = torch.cat([   crop(image,  mask,  0.50),
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
                out = F.softmax(net(tmp), dim=-1).mean(dim=0)/cfg.rate
                out = str(out.cpu().numpy().argmax())
                f.write(imgID[0]+','+out+'\n')
                cnt += 1
                print(cnt)


if __name__=='__main__':
    test(dataset, network)
