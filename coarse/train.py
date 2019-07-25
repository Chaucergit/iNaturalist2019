#!/usr/bin/python3
#coding=utf-8

import os
import sys
import math
import datetime
from apex import amp
sys.path.insert(0, '../')
sys.dont_write_bytecode = True

import apex
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

import dataset
import network

def train(Dataset, Network):
    ## dataset
    cfg    = Dataset.Config(savepath='./out', snapshot=None, mode='train', batch=14, lr=0.04, momen=0.9, decay=5e-4, epoch=32)
    data   = Dataset.Data(cfg)
    loader = DataLoader(data, batch_size=cfg.batch, shuffle=True, num_workers=32, pin_memory=True)
    ## network
    net    = Network.Classify(cfg)
    net.train(True)
    net.cuda()
    ## parameter
    base, head = [], []
    for name, param in net.named_parameters():
        if 'layer0' in name or 'layer1' in name:
            param.requires_grad = False
        elif 'bkbone' in name:
            base.append(param)
        else:
            head.append(param)

    optimizer      = torch.optim.SGD([{'params':base}, {'params':head}], lr=cfg.lr, momentum=cfg.momen, weight_decay=cfg.decay, nesterov=True)
    net, optimizer = amp.initialize(net, optimizer, opt_level='O2')
    sw             = SummaryWriter('./log')
    global_step    = 0

    for epoch in range(cfg.epoch):
        optimizer.param_groups[0]['lr'] = 0.1*cfg.lr*(1-abs((1+epoch)/(1+cfg.epoch)*2-1))
        optimizer.param_groups[1]['lr'] =     cfg.lr*(1-abs((1+epoch)/(1+cfg.epoch)*2-1))

        for step, (image, label) in enumerate(loader):
            ## forward
            image, label = image.cuda().float(), label.cuda().long()
            out          = net(image).mean(dim=(2,3))
            loss         = F.cross_entropy(out, label)
            optimizer.zero_grad()
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
            optimizer.step()

            ## log
            if step%10 == 0:
                global_step += 10
                _, idx       = torch.sort(out, dim=1, descending=True)
                top1         = (idx[:,:1] == label.unsqueeze(1)).sum().item()/out.size(0)
                top2         = (idx[:,:2] == label.unsqueeze(1)).sum().item()/out.size(0)
                top3         = (idx[:,:3] == label.unsqueeze(1)).sum().item()/out.size(0)
                top4         = (idx[:,:4] == label.unsqueeze(1)).sum().item()/out.size(0)
                top5         = (idx[:,:5] == label.unsqueeze(1)).sum().item()/out.size(0)
                sw.add_scalars('hyper', {'body':optimizer.param_groups[0]['lr'], 'head':optimizer.param_groups[1]['lr']}, global_step=global_step)
                sw.add_scalars('loss' , {'loss':loss.item()}, global_step=global_step)
                sw.add_scalars('accu' , {'top1':top1, 'top2':top2, 'top3':top3, 'top4':top4, 'top5':top5}, global_step=global_step)
                print('%s | step:%d/%d/%d | lr=%.6f | loss=%.6f | top1=%.6f | top2=%.6f | top3=%.6f | top4=%.6f | top5=%.6f'
                        %( datetime.datetime.now(),  global_step, epoch+1, cfg.epoch, optimizer.param_groups[1]['lr'], loss.item(), top1, top2, top3, top4, top5))

        if not os.path.exists(cfg.savepath):
            os.makedirs(cfg.savepath)
        torch.save(net.state_dict(), cfg.savepath+'/model-'+str(epoch+1))

if __name__=='__main__':
    train(dataset, network)

