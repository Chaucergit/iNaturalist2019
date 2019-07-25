#!/usr/bin/python3
#coding=utf-8

import os
import cv2
import json
import torch
import numpy as np
import transform
from torch.utils.data import Dataset

class Config(object):
    def __init__(self, **kwargs):
        self.kwargs       = kwargs
        self.mean         = np.array([[[115.75, 120.83, 93.49]]])
        self.std          = np.array([[[ 50.46,  50.08, 49.67]]])
        self.cls_beg      = 0
        self.cls_end      = 1010

        ## different mode
        if self.mode == 'train':
            self.train()
        elif self.mode == 'val':
            self.val()
        elif self.mode == 'test':
            self.test()
        else:
            raise ValueError

        ## each class rate
        with open('../data/info.txt', 'r') as lines:
            self.rate  = torch.zeros(self.cls_end-self.cls_beg).cuda().float()
            for line in lines:
                cls, cnt = line.strip().split(',')
                cls, cnt = int(cls), int(cnt)
                if cls >= self.cls_beg and cls < self.cls_end:
                    self.rate[cls-self.cls_beg] = cnt

        ## print hyper parameters
        print('\nParameters...')
        for k, v in self.kwargs.items():
            print('%-10s: %s'%(k, v))
        print('%-10s: %s'%("sample", len(self.samples)))

    def train(self):
        with open('../data/train.txt', 'r') as lines:
            self.samples = []
            for line in lines:
                name, imgID, cls = line.strip().split(',')
                path = '../data/train/'+cls+'/'+name
                cls  = int(cls)
                if cls >= self.cls_beg and cls < self.cls_end:
                    self.samples.append([path, cls-self.cls_beg])

    def val(self):
        with open('../data/val.txt', 'r') as lines:
            self.samples = []
            for line in lines:
                name, imgID, cls = line.strip().split(',')
                path = '../data/train/'+cls+'/'+name
                cls  = int(cls)
                if cls >= self.cls_beg and cls < self.cls_end:
                    self.samples.append([path, cls-self.cls_beg])

    def test(self):
        with open('../data/test.txt', 'r') as lines:
            self.samples = []
            for line in lines:
                name, imgID = line.strip().split(',')
                path = '../data/test/'+name
                self.samples.append([path, imgID])

    def __getattr__(self, name):
        if name in self.kwargs:
            return self.kwargs[name]
        else:
            return None


class Data(Dataset):
    def __init__(self, cfg):
        self.cfg = cfg
        if self.cfg.mode == 'train':
            self.transform = transform.Compose( transform.Normalize(mean=cfg.mean, std=cfg.std),
                                                transform.Resize(size=512),
                                                transform.RandomRotate(-15, 15),
                                                transform.RandomCrop(448, 448),
                                                transform.RandomHorizontalFlip(),
                                                transform.RandomMask(),
                                                transform.ToTensor())
        elif self.cfg.mode == 'val' or self.cfg.mode == 'test': 
            self.transform = transform.Compose( transform.Normalize(mean=cfg.mean, std=cfg.std),
                                                transform.Resize(size=512),
                                                transform.ToTensor())
        else:
            raise ValueError

    def __getitem__(self, idx):
        if self.cfg.mode == 'train':
            path, label   = self.cfg.samples[idx]
            image         = cv2.imread(path).astype(np.float32)[:,:,::-1]
            image         = self.transform(image)
            return image, label
        elif self.cfg.mode == 'val':
            path, label   = self.cfg.samples[idx]
            image         = cv2.imread(path).astype(np.float32)[:,:,::-1]
            image         = self.transform(image)
            return image, label
        elif self.cfg.mode == 'test':
            path, imgID = self.cfg.samples[idx]
            image       = cv2.imread(path).astype(np.float32)[:,:,::-1]
            image       = self.transform(image)
            return image, imgID

    def __len__(self):
        return len(self.cfg.samples)

if __name__=='__main__':
    import matplotlib.pyplot as plt 

    cfg    = Config(mode='train')
    data   = Data(cfg)
    for i in range(1000):
        image, label = data[i]
        print(label)
        image = image.permute(1,2,0).numpy()*cfg.std+cfg.mean
        plt.imshow(np.uint8(image))
        plt.show()
