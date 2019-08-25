#!/usr/bin/env python
# coding=utf-8
import torch
import torch.nn as nn
'''
### for x = (b, c, t, h, w)
class TemporalAttention(nn.Module):
    def __init__(self, temporal, reduction=1, multiply=True):
        super(TemporalAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
                nn.Linear(temporal, temporal // reduction),
                nn.ReLU(inplace=True),
                nn.Linear(temporal // reduction, temporal),
                nn.Sigmoid()
        )
        self.multiply=multiply

    def forward(self, x):
        #x.size() (b, c, t, h, w)
        b, t, _, _, _ = torch.transpose(x, 1, 2).size() # (b, t, c, h, w)
        out = self.avg_pool(torch.transpose(x, 1, 2)).view(b, t)
        w = self.fc(out).view(b, t, 1, 1, 1) # (b, 1, t, 1, 1)
        w = torch.transpose(w, 1, 2)
        if self.multiply == True:
            return x * w
        else:
            return w

'''
### for x = (b, c, t, 1, 1)->view-> (b, c, t)
class TemporalAttention(nn.Module): 
    def __init__(self, temporal, reduction=1, multiply=True):
        super(TemporalAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
                nn.Linear(temporal, temporal // reduction),
                nn.ReLU(inplace=True),
                nn.Linear(temporal // reduction, temporal),
                nn.Sigmoid()
        )
        self.multiply=multiply

    def forward(self, x):
        #x.size() (b, c, t)
        b, t, _ = torch.transpose(x, 1, 2).size() # (b, t, c)
        out = self.avg_pool(torch.transpose(x, 1, 2)).view(b,t)
        w = self.fc(out).view(b, t, 1)# (b, t, 1)
        w = torch.transpose(w, 1 ,2) # (b, 1, t) 0~1
        if self.multiply == True:
            return x * w
        else:
            return w

class TemporalMaxPool(nn.Module):
    def __init__(self):
        super(TemporalMaxPool, self).__init__()
        self.filter=nn.AdaptiveMaxPool1d(1)

    def forward(self, x):
        out=self.filter(x)
        out=torch.squeeze(out)
        return out

class TemporalAvgPool(nn.Module):
    def __init__(self):
        super(TemporalAvgPool, self).__init__()
        self.filter=nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        out=self.filter(x)
        out=torch.squeeze(out)
        return out

class AttentionLayer(nn.Module): # (b,c,h,w) for channel
    def __init__(self, channel, reduction=64, multiply=True):
        super(AttentionLayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
                nn.Linear(channel, channel // reduction),
                nn.ReLU(inplace=True),
                nn.Linear(channel // reduction, channel),
                nn.Sigmoid()
        )
        self.multiply = multiply

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        if self.multiply == True:
            return x * y
        else:
            return y


