import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import sys, os

from .cc_attention import CrissCrossAttention
from models.map_modules import get_padded_mask_and_weight

class MapConv(nn.Module):

    def __init__(self, cfg):
        super(MapConv, self).__init__()
        input_size = cfg.INPUT_SIZE
        hidden_sizes = cfg.HIDDEN_SIZES
        kernel_sizes = cfg.KERNEL_SIZES
        strides = cfg.STRIDES
        paddings = cfg.PADDINGS
        dilations = cfg.DILATIONS
        self.convs = nn.ModuleList()
        assert len(hidden_sizes) == len(kernel_sizes) \
               and len(hidden_sizes) == len(strides) \
               and len(hidden_sizes) == len(paddings) \
               and len(hidden_sizes) == len(dilations)
        channel_sizes = [input_size]+hidden_sizes
        for i, (k, s, p, d) in enumerate(zip(kernel_sizes, strides, paddings, dilations)):
            self.convs.append(nn.Conv2d(channel_sizes[i], channel_sizes[i+1], k, s, p, d))

    def forward(self, x, mask):
        padded_mask = mask
        for i, pred in enumerate(self.convs):
            x = F.relu(pred(x))
            padded_mask, masked_weight = get_padded_mask_and_weight(padded_mask, pred)
            x = x * masked_weight
        return x


class GELU(nn.Module):

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))


class RCCABlock(nn.Module):
    def __init__(self, cfg):
        super(RCCABlock, self).__init__()
        in_channels = cfg.INPUT_CHANNEL
        inter_channels = in_channels // 4    
        out_channels = cfg.OUTPUT_CHANNEL
        self.recurrence = cfg.LOOP_NUM
        self.conva = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
                                   nn.ReLU(inplace=True),)
        self.cca = CrissCrossAttention(inter_channels)
        self.convb = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),
                                   nn.ReLU(inplace=True))

        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels+inter_channels, out_channels, kernel_size=3, padding=1, dilation=1, bias=False),
            nn.ReLU(inplace=True),
            )

    def forward(self, x, map_mask):
        map_mask = map_mask.float()
        output = self.conva(x)
        for i in range(self.recurrence):
            output = self.cca(output)
        output = self.convb(output)

        output = self.bottleneck(torch.cat([x, output], 1)) * map_mask
        return output


class RCCAModule(nn.Module):
    def __init__(self, cfg):
        super(RCCAModule, self).__init__()
        self.block_num = cfg.RCCA_NUM
        self.rcca_block = nn.ModuleList([RCCABlock(cfg) for _ in range(self.block_num)])

    def forward(self, x, map_mask):
        for i in range(len(self.rcca_block)):
            x = self.rcca_block[i](x, map_mask)
        return x

