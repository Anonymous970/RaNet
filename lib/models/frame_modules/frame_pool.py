import torch
from torch import nn
import math
import numpy as np


class PositionalEmbedding(nn.Module):

    def __init__(self, d_model, max_len=256):
        super().__init__()

        pe = torch.zeros(d_model, max_len).float()   
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[0::2, :] = torch.sin(position * div_term)
        pe[1::2, :] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)   
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :, :x.size(2)]


class LearnPositionalEncoding(nn.Module):

    def __init__(self, d_model, max_len=64, dropout=0.1):
        super(LearnPositionalEncoding, self).__init__()
        self.pos_embed = nn.Embedding(max_len, d_model)

        nn.init.uniform_(self.pos_embed.weight)

        self.dropout = nn.Dropout(p=dropout)


    def forward(self, q):
        bsz_q, d_model, q_frm = q.shape
        assert q_frm == self.pos_embed.weight.shape[0], (q_frm,self.pos_embed.weight.shape)
        q_pos = self.pos_embed.weight.clone()
        q_pos = q_pos.unsqueeze(0)
        q_pos = q_pos.expand(bsz_q, q_frm, d_model).transpose(1,2)
        # q_pos = q_pos.contiguous().view(bsz_q, q_frm, n_head, d_k)
        q = q + q_pos
        return self.dropout(q)


class FrameAvgPool(nn.Module):

    def __init__(self, cfg):
        super(FrameAvgPool, self).__init__()
        input_size = cfg.INPUT_SIZE
        hidden_size = cfg.HIDDEN_SIZE
        kernel_size = cfg.KERNEL_SIZE
        stride = cfg.STRIDE
        self.vis_conv = nn.Conv1d(input_size, hidden_size, 1, 1)
        self.avg_pool = nn.AvgPool1d(kernel_size, stride)

        if cfg.USE_POSITION:
            self.pos_embed = LearnPositionalEncoding(d_model=hidden_size, max_len=cfg.NUM_CLIPS)
        else:
            self.pos_embed = None

    def forward(self, visual_input):
        vis_h = torch.relu(self.vis_conv(visual_input))
        vis_h = self.avg_pool(vis_h)
        if self.pos_embed:
            vis_h = self.pos_embed(vis_h) 
        return vis_h




