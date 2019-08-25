#!/usr/bin/env python
# coding=utf-8
import torch 
import torch.nn as nn

class MaxEntropyLoss(nn.Module):
    def __init__(self):
        super(MaxEntropyLoss,self).__init__()
    
    def forward(self,x):
        # x (b,1,t) 0~1
        x=torch.squeeze(x)
        #x (b,t) 0~1
        #loss=torch.mean((torch.sum(x,1)/x.size(1)-0.5)**2)
        loss=torch.mean((torch.sum(x,0)/x.size(0)-0.5)**2)
        return loss
        


