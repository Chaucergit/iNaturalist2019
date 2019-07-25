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
                                                transform.Resize(size=448),
                                                transform.RandomHorizontalFlip(),
                                                transform.ToTensor())
        elif self.cfg.mode == 'test' or self.cfg.mode == 'val':
            self.transform = transform.Compose( transform.Normalize(mean=cfg.mean, std=cfg.std),
                                                transform.ToTensor())
        else:
            raise ValueError


    def crop(self, image, mask):
        H, W  = mask.shape
        S     = int(np.sqrt(mask.sum()*2))
        X, Y  = np.where(mask==1)
        if len(X)<10 or len(Y)<10 or S<10:
            return image
        randx = np.random.randint(len(X)//3, len(X)//3*2)
        randy = np.random.randint(len(Y)//3, len(Y)//3*2)
        xmin  = max(int(X[randx]-S/2), 0)
        ymin  = max(int(Y[randy]-S/2), 0)
        xmax  = min(int(X[randx]+S/2), H)
        ymax  = min(int(Y[randy]+S/2), W)
        image = image[xmin:xmax,ymin:ymax,:]
        return image

    def __getitem__(self, idx):
        if self.cfg.mode == 'train':
            path, label   = self.cfg.samples[idx]
            image         = cv2.imread(path).astype(np.float32)[:,:,::-1]
            path          = path.replace('train', 'train-mask').replace('.jpg', '.png')
            mask          = cv2.imread(path, -1).astype(np.float32)/255.0
            image         = self.crop(image, mask)
            image         = self.transform(image)
            return image, label
        elif self.cfg.mode == 'val':
            path, label   = self.cfg.samples[idx]
            image         = cv2.imread(path).astype(np.float32)[:,:,::-1]
            path          = path.replace('train', 'val-mask').replace('.jpg', '.png')
            mask          = cv2.imread(path, -1).astype(np.float32)/255.0
            image         = self.transform(image)
            return image, label, mask
        elif self.cfg.mode == 'test':
            path, imgID = self.cfg.samples[idx]
            image       = cv2.imread(path).astype(np.float32)[:,:,::-1]
            path        = path.replace('test', 'test-mask').replace('.jpg', '.png')
            mask        = cv2.imread(path, -1).astype(np.float32)/255.0
            image       = self.transform(image)
            return image, imgID, mask

    def __len__(self):
        return len(self.cfg.samples)

if __name__=='__main__':
    import matplotlib.pyplot as plt 

    cfg    = Config(mode='train')
    data   = Data(cfg)
    for i in range(1000):
        image, label, mask = data[i]
        print(label)
        image = image.permute(1,2,0).numpy()*cfg.std+cfg.mean
        plt.subplot(121)
        plt.imshow(np.uint8(image))
        plt.subplot(122)
        plt.imshow(mask)
        plt.show()
