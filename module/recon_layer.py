import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.utils import spectral_norm

from recon_utils import src_mask
from .utils import *


class MultiHeadAttention(nn.Module):
    def __init__(self, config, d_hid, decoder=False, conditional=False):
        super().__init__()

        d_head = config["Model"]["Reconstruction"]["d_head"]
        d_spk = config["Model"]["Reconstruction"]["d_speaker"]

        dropout = config["Model"]["Reconstruction"]["dropout"]

        self.attn = nn.MultiheadAttention(d_hid, d_head)
        self.dropout = nn.Dropout(dropout)
        self.norm = ConvLayerNorm(d_hid, d_spk) if conditional else nn.LayerNorm(d_hid)

        self.decoder = decoder # for decide to mask the attention weights.

    def forward(self, query, key=None, cond=None):
        if key is None:
            key = query
        
        tot_timeStep = query.shape[1]       # (B, T, C)
        
        query = query.contiguous().transpose(0, 1)
        key = key.contiguous().transpose(0, 1)

        if self.decoder:
            query = self.attn(query, key, key, attn_mask = src_mask(tot_timeStep).to(query.device))[0]
        else:
            query = self.attn(query, key, key)[0]
        query = query.contiguous().transpose(0, 1)

        if cond is not None:
            return self.norm(self.dropout(query) + query, cond)
        else:
            return self.norm(self.dropout(query) + query)


class ConvFeedForward(nn.Module):
    def __init__(self, config, d_hid, kernel_size=None, conditional=False):
        super().__init__()

        """ Parameter """
        d_spk = config["Model"]["Reconstruction"]["d_speaker"]
        dropout = config["Model"]["Reconstruction"]["dropout"]

        if kernel_size is None:
            kernel_size = config["Model"]["Reconstruction"]["kernel_size"]
        padding = (kernel_size - 1) // 2

        """ Layer """
        self.conv1 = nn.Conv1d(d_hid, 2 * d_hid, kernel_size = kernel_size, padding = padding)
        self.conv2 = nn.Conv1d(2 * d_hid, d_hid, kernel_size = kernel_size, padding = padding)

        self.dropout = nn.Dropout(dropout)
        self.norm = ConvLayerNorm(d_hid, d_spk) if conditional else nn.LayerNorm(d_hid)
        

    def forward(self, x, cond=None):
        """ (B, T, C) -> (B, T, C) """
        out = x.contiguous().transpose(1, 2)

        out = self.conv2(F.gelu(self.conv1(out)))
        out = out.contiguous().transpose(1, 2)

        if cond is not None:
            out = self.norm(self.dropout(out) + x, cond)
        else:
            out = self.norm(self.dropout(out) + x)
        return out


class LinearFeedForward(nn.Module):
    def __init__(self, config, d_hid, spec_norm=True):
        super().__init__()

        """ Parameter """
        dropout = config["Model"]["Reconstruction"]["dropout"]

        """ Layer """
        f = spectral_norm if spec_norm else lambda x: x
        self.linear1 = f(nn.Linear(d_hid, 2 * d_hid))
        self.linear2 = f(nn.Linear(2 * d_hid, d_hid))

        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_hid)
        

    def forward(self, x, cond=None):
        """ (B, T, C) -> (B, T, C) """

        out = self.linear2(F.gelu(self.linear1(x)))
        out = self.norm(self.dropout(out) + x)
        return out



""" Attnetion Blocks """

class EncAttnBlock(nn.Module):
    def __init__(self, config):
        super().__init__()

        d_hid = config['Model']['Reconstruction']['d_hid_encoder']

        """ Architecture """
        self.attn = MultiHeadAttention(config, d_hid)
        self.conv = LinearFeedForward(config, d_hid, spec_norm=False)


    def forward(self, x):
        """ 
        ? INPUT
        - x: (B, T+1, C)
            Speaker Encoder is first vector on time axis.

        ? Output
        - output: (B, T+1, C)
        """

        return self.conv(self.attn(x))


class EncConvBlock(nn.Module):
    def __init__(self, config):
        super().__init__()

        """ Parameter """
        d_hid = config["Model"]["Reconstruction"]["d_hid_encoder"]
        scale_factor = config["Model"]["Reconstruction"]["scale_factor"]
        
        """ Architecture """
        self.shuffle = PixelShuffle(scale_factor)
        self.downsample = nn.AvgPool1d(kernel_size=scale_factor, stride=scale_factor)

        self.conv1 = Conv(config, d_hid, d_hid)
        self.conv2 = Conv(config, d_hid, d_hid)

    def forward(self, x):
        # hid = self.shuffle(self.conv1(x))
        # hid = self.conv2(hid)
        # return self.downsample(x.contiguous().transpose(1, 2)).contiguous().transpose(1, 2) + hid

        return x + self.conv2(self.conv1(x))



class DecAttnBlock(nn.Module):
    def __init__(self, config):
        super().__init__()

        d_hid = config['Model']['Reconstruction']['d_hid_decoder']

        """ Architecture """
        self.attn = MultiHeadAttention(config, d_hid, decoder=False, conditional=True)
        self.conv = ConvFeedForward(config, d_hid, conditional=True)

    def forward(self, x, cond):
        return self.conv(self.attn(x, x, cond), cond)



class DecConvBlock(nn.Module):
    def __init__(self, config):
        super().__init__()

        """ Parameter """
        d_hid = config["Model"]["Reconstruction"]["d_hid_decoder"]
        scale_factor = config["Model"]["Reconstruction"]["scale_factor"]
        self.scale_factor = scale_factor
        
        """ Architecture """
        self.shuffle = InversePixelShuffle(scale_factor)
        self.downsample = nn.AvgPool1d(kernel_size=scale_factor, stride=scale_factor)

        self.conv1 = Conv(config, d_hid, d_hid, decoder=True)
        self.conv2 = Conv(config, d_hid, d_hid, decoder=True)

    def forward(self, x, cond):
        # hid = self.shuffle(self.conv1(x, cond))                           # (B, 2*T, C//2)
        # hid = self.conv2(hid, cond)   # (B, 2*T, C)

        # res = F.interpolate(x.contiguous().transpose(1, 2), 
        #     scale_factor=self.scale_factor, mode='nearest').contiguous().transpose(1, 2)
        # return res + hid

        return x + self.conv2(self.conv1(x, cond), cond)







class Conv(nn.Module):
    def __init__(self, config, d_in, d_out, decoder=False):
        super().__init__()

        self.decoder = decoder
        if decoder:
            d_spk = config['Model']['Reconstruction']['d_speaker']

        dropout = config['Model']['Reconstruction']['dropout']
        kernel_size = config["Model"]['Reconstruction']['kernel_size']
        padding = (kernel_size - 1) // 2

        """ Architecture """
        #self.conv = nn.Conv1d(d_hid, d_hid, kernel_size, padding=padding)
        self.conv = nn.Conv1d(d_in, d_out, kernel_size, padding=padding, padding_mode='reflect')
        self.dropout = nn.Dropout(dropout)
        self.norm = ConvLayerNorm(d_out, d_spk) if decoder else nn.LayerNorm(d_out)


    def forward(self, x, cond=None):
        out = self.dropout(self.conv(x.contiguous().transpose(1, 2)).contiguous().transpose(1, 2))
        if self.decoder:
            out = F.gelu(self.norm(out, cond))
        else:
            out = F.gelu(self.norm(out))

        return out



