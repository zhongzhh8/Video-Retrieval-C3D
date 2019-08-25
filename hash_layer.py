#!/usr/bin/env python
# coding=utf-8
import torch
import torch.nn as nn
class HashLayer(nn.Module):
    def __init__(self,hash_length,num_class):
        super(HashLayer, self).__init__()
        self.hashcoder=nn.Sequential(nn.Linear(512,hash_length), nn.Tanh())
        self.classifier=nn.Linear(512,num_class)

    def forward(self,x):
        if x.size()==5:
            x=x.view()
        h=self.hashcoder(x)
        c=self.classifier(x)
        return h, c

