#!/usr/bin/python3
#coding=utf-8

import cv2
import torch
import numpy as np

class Compose(object):
    def __init__(self, *ops):
        self.ops = ops
    
    def __call__(self, image):
        for op in self.ops:
            image = op(image)
        return image

class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean 
        self.std  = std
    
    def __call__(self, image):
        image = (image - self.mean)/self.std
        return image

class Resize(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, image):
        ## random resize
        H,W,_ = image.shape
        image = cv2.resize(image, dsize=(self.size, self.size), interpolation=cv2.INTER_LINEAR)
        return image

class CenterCrop(object):
    def __init__(self, H, W):
        self.H = H
        self.W = W
    
    def __call__(self, image):
        H,W,_ = image.shape
        xmin  = int((W-self.W)/2)
        ymin  = int((H-self.H)/2)
        image = image[ymin:ymin+self.H, xmin:xmin+self.W, :]
        return image

class RandomHorizontalFlip(object):
    def __call__(self, image):
        if np.random.randint(2)==1:
            image = image[:,::-1,:].copy()
        return image

class RandomVerticalFlip(object):
    def __call__(self, image):
        if np.random.randint(2)==1:
            image = image[::-1,:,:].copy()
        return image

class RandomBlur(object):
    def __call__(self, image):
        if np.random.randint(2)==1:
            image = cv2.blur(image, ksize=(5, 5))
        return image

class RandomNoise(object):
    def __call__(self, image):
        if np.random.randint(2)==1:
            noise = np.random.normal(0, 0.01, size=image.shape)
            image = image + noise
        return image

class RandomBright(object):
    def __call__(self, image):
        image = np.random.uniform(low=0.9, high=1.1)*image
        return image

class RandomRotate(object):
    def __init__(self, angle1, angle2):
        self.angle1 = angle1
        self.angle2 = angle2

    def __call__(self, image):
        H, W, C = image.shape
        angle   = np.random.randint(self.angle1, self.angle2)
        M       = cv2.getRotationMatrix2D(center=(W//2, H//2), angle=angle, scale=1.0)
        image   = cv2.warpAffine(image, M, (W, H))
        return image

class RandomMask(object):
    def __call__(self, image):
        mask =  np.random.binomial(1, 0.9, (4,4))
        mask = cv2.resize(mask, dsize=image.shape[:-1], interpolation=cv2.INTER_NEAREST)[:,:,np.newaxis]
        return mask*image

class ToTensor(object):
    def __call__(self, image):
        image = torch.from_numpy(image)
        image = image.permute(2, 0, 1)
        return image
