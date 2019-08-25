#!/usr/bin/env python
# coding=utf-8
import torch
import torch.nn as nn


class QuantizationLoss(nn.Module):
    def __init__(self):
        super(QuantizationLoss, self).__init__()

    def forward(self, x):
        # x: b*1*t
        x = torch.squeeze(x)  # b*t 0~1
        # loss=torch.mean(torch.sum((0.5-torch.abs(x-0.5))**2,1)/x.size(1))
        loss = torch.mean(torch.sum((0.5 - torch.abs(x - 0.5)), 1) / x.size(1))

        return loss

