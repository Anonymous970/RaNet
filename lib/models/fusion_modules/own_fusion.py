import torch
from torch import nn
import torch.nn.functional as F
import math
import numpy as np
import pdb
from torch.autograd import Variable


class LSTMMaxPoolDot(nn.Module):

    def __init__(self, cfg):
        super(LSTMMaxPoolDot, self).__init__()
        self.cfg = cfg
        self.textual_encoder = nn.LSTM(cfg.TXT_INPUT_SIZE, cfg.TXT_HIDDEN_SIZE//2 if cfg.LSTM.BIDIRECTIONAL else cfg.TXT_HIDDEN_SIZE,
                                       num_layers=cfg.LSTM.NUM_LAYERS, bidirectional=cfg.LSTM.BIDIRECTIONAL, batch_first=True)
        self.tex_linear = nn.Linear(cfg.TXT_HIDDEN_SIZE, cfg.HIDDEN_SIZE)
        self.vis_conv = nn.Conv2d(cfg.VIS_INPUT_SIZE, cfg.HIDDEN_SIZE, 1, 1)

    def forward(self, textual_input, textual_mask, map_h):
        self.textual_encoder.flatten_parameters()
        txt_h = self.textual_encoder(textual_input)[0] * textual_mask
        txt_h = torch.max(txt_h, dim=1)[0]
        txt_h = self.tex_linear(txt_h)[:,:,None,None]  # 4, 512, 1, 1
        map_h = self.vis_conv(map_h)  # 4, 512, 128, 128
        fused_h = F.normalize(txt_h * map_h)
        return fused_h


class DynamicFuse(nn.Module):
    def __init__(self, cfg):
        super(DynamicFuse, self).__init__()

        self.cfg = cfg
        self.textual_encoder = nn.LSTM(cfg.TXT_INPUT_SIZE, cfg.TXT_HIDDEN_SIZE//2 if cfg.LSTM.BIDIRECTIONAL else cfg.TXT_HIDDEN_SIZE,
                                       num_layers=cfg.LSTM.NUM_LAYERS, bidirectional=cfg.LSTM.BIDIRECTIONAL, batch_first=True)
        self.tex_linear_b1 = nn.Linear(cfg.TXT_HIDDEN_SIZE, cfg.HIDDEN_SIZE)
        self.vis_conv_b1 = nn.Conv2d(cfg.VIS_INPUT_SIZE, cfg.HIDDEN_SIZE, 1, 1)
        self.tex_linear_b2_a = nn.Linear(cfg.TXT_HIDDEN_SIZE, cfg.HIDDEN_SIZE)
        self.vis_conv_b2_a = nn.Conv2d(cfg.VIS_INPUT_SIZE, cfg.HIDDEN_SIZE, 1, 1)
        self.tex_linear_b2_b = nn.Linear(cfg.TXT_HIDDEN_SIZE, cfg.HIDDEN_SIZE)
        self.vis_conv_b2_b = nn.Conv2d(cfg.VIS_INPUT_SIZE, cfg.HIDDEN_SIZE, 1, 1)
        self.softmax = nn.Softmax(dim=1)
        self.relu = nn.ReLU(True)

    def forward(self, textual_input, textual_mask, map_h):
        self.textual_encoder.flatten_parameters()
        txt_h = self.textual_encoder(textual_input)[0] * textual_mask  # B, L, C
        txt_pool = torch.max(txt_h, dim=1)[0]  # B, C
        txt_h_b1 = self.tex_linear_b1(txt_pool)[:,:,None,None]  # B, C, 1, 1
        map_h_b1 = self.vis_conv_b1(map_h)  # B ,C, T, T
        fused_b1 = F.normalize(txt_h_b1 * map_h_b1)

        txt_h_b2_a = self.tex_linear_b2_a(txt_h)  # B, L, C
        map_h_b2_a = self.vis_conv_b2_a(map_h)  # B, C, T, T
        fuse_mask = self.softmax(torch.matmul(txt_h_b2_a, map_h_b2_a.view(map_h_b2_a.size(0), map_h_b2_a.size(1), -1)))  # B, L, T*T
        txt_h_b2_b = self.tex_linear_b2_b(txt_h)  # B, L, C
        map_h_b2_b = self.vis_conv_b2_b(map_h)  # B, C, T, T
        txt_attn = torch.matmul(txt_h_b2_b.transpose(-1, -2), fuse_mask).view(map_h_b2_b.size(0), -1, map_h_b2_b.size(2), map_h_b2_b.size(3))  # B, C, T, T
        fused_b2 = F.normalize(txt_attn * map_h_b2_b)

        fused_h = self.relu(fused_b1 + fused_b2)
        return fused_h
