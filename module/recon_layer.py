import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.utils import spectral_norm

from recon_utils import src_mask


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
    def __init__(self, config, d_hid, conditional=False):
        super().__init__()

        """ Parameter """
        dropout = config["Model"]["Reconstruction"]["dropout"]

        """ Layer """
        f = spectral_norm
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

class Enc_AttnBlock(nn.Module):
    def __init__(self, config):
        super().__init__()

        d_hid = config['Model']['Reconstruction']['d_hid_encoder']

        """ Architecture """
        self.attn = MultiHeadAttention(config, d_hid)
        self.conv = ConvFeedForward(config, d_hid)


    def forward(self, x):
        """ 
        ? INPUT
        - x: (B, T+1, C)
            Speaker Encoder is first vector on time axis.

        ? Output
        - output: (B, T+1, C)
        """

        return self.conv(self.attn(x))



class Dec_AttnBlock(nn.Module):
    def __init__(self, config):
        super().__init__()

        d_hid = config['Model']['Reconstruction']['d_hid_decoder']

        """ Architecture """
        self.attn = MultiHeadAttention(config, d_hid, decoder=False, conditional=True)
        self.conv = ConvFeedForward(config, d_hid, conditional=True)

    def forward(self, x, cond):
        return self.conv(self.attn(x, x, cond), cond)


class DecoderConv(nn.Module):
    def __init__(self, config):
        super().__init__()

        d_hid = config['Model']['Reconstruction']['d_hid_decoder']
        d_spk = config['Model']['Reconstruction']['d_speaker']

        dropout = config['Model']['Reconstruction']['dropout']
        kernel_size = config["Model"]['Reconstruction']['kernel_size']
        padding = (kernel_size - 1) // 2
        self.stride = 2

        """ Architecture """
        #self.conv = nn.Conv1d(d_hid, d_hid, kernel_size, padding=padding)
        self.conv = nn.Conv1d(d_hid, d_hid, kernel_size, padding=padding)
        self.dropout = nn.Dropout(dropout)
        self.norm = ConvLayerNorm(d_hid, d_spk)


    def forward(self, x, cond):
        out = self.dropout(self.conv(x.contiguous().transpose(1, 2)).contiguous().transpose(1, 2))
        out = F.gelu(self.norm(out, cond))

        return out


class ContentsConv(nn.Module):
    def __init__(self, config, d_in, d_out):
        super().__init__()

        d_hid = config['Model']['Reconstruction']['d_hid_encoder']
        d_cnt = config['Model']['Reconstruction']['d_contents']

        dropout = config['Model']['Reconstruction']['dropout']
        kernel_size = config["Model"]['Reconstruction']['kernel_size']
        padding = (kernel_size - 1) // 2

        """ Architecture """
        self.conv = nn.Conv1d(d_in, d_out, kernel_size = kernel_size, padding = padding)
        #self.conv = nn.Conv1d(d_in, d_out, kernel_size, padding = padding, stride=2)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_out)

    def forward(self, x):
        x = self.conv(x.contiguous().transpose(1, 2)).contiguous().transpose(1, 2)
        return F.gelu(self.norm(self.dropout(x)))






""" Utils """

class ConvLayerNorm(nn.Module):
    def __init__(self, d_hid, d_cond=None):
        super().__init__()

        if d_cond is None:
            d_cond = 128

        self.linear = nn.Linear(d_cond, 2 * d_hid)

    def forward(self, x, cond):
        """
        ? INPUT
            - x: (B, T, C)
            - cond: (B, 1, C)
        ? OUTPUT
            - (B, T, C), torch
        """
        if len(cond.shape) == 2:
            cond = cond.unsqueeze(1)
        
        scale, bias = self.linear(cond).chunk(2, dim=-1)
        return layer_norm(x, dim=1) * scale + bias
        
def layer_norm(x, dim, eps: float = 1e-14):
    mean = torch.mean(x, dim=dim, keepdim=True)
    var = torch.square(x - mean).mean(dim=dim, keepdim=True)
    return (x - mean) / torch.sqrt(var + eps)




""" From StyleGAN """
from math import sqrt
# Reference: https://github.com/rosinality/style-based-gan-pytorch/blob/master/model.py

class EqualLR:
    def __init__(self, name):
        self.name = name

    def compute_weight(self, module):
        weight = getattr(module, self.name + '_orig')
        fan_in = weight.data.size(1) * weight.data[0][0].numel()

        return weight * sqrt(2 / fan_in)

    @staticmethod
    def apply(module, name):
        fn = EqualLR(name)

        weight = getattr(module, name)
        del module._parameters[name]
        module.register_parameter(name + '_orig', nn.Parameter(weight.data))
        module.register_forward_pre_hook(fn)

        return fn

    def __call__(self, module, input):
        weight = self.compute_weight(module)
        setattr(module, self.name, weight)


def equal_lr(module, name='weight'):
    EqualLR.apply(module, name)

    return module


class PixelNormLayer(nn.Module):
    def __init__(self, epsilon=1e-8):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, x):
        return x * torch.rsqrt(torch.mean(x ** 2, dim=1, keepdim=True) + self.epsilon)


class EqualLinear(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()

        linear = nn.Linear(in_dim, out_dim)
        linear.weight.data.normal_()
        linear.bias.data.zero_()

        self.linear = equal_lr(linear)

    def forward(self, input):
        return self.linear(input)
