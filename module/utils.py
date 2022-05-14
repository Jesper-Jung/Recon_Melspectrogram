import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.utils import spectral_norm

from recon_utils import src_mask


""" Utils """
# https://github.com/cyhuang-tw/AdaIN-VC/blob/88fe7338b27698d31855bec96ee4874e5027f1a2/model.py
class PixelShuffle(nn.Module):
    def __init__(self, scale_factor: int):
        super().__init__()
        self.scale_factor = scale_factor

    def forward(self, x):
        """ Downsampling along frequency-axis + Upsampling along time-axis """

        batch_size, channels, in_width = x.size()
        channels = channels // self.scale_factor
        out_width = in_width * self.scale_factor

        x = x.contiguous().view(batch_size, channels, self.scale_factor, in_width)
        x = x.permute(0, 1, 3, 2).contiguous()
        x = x.view(batch_size, channels, out_width)
        return x

class InversePixelShuffle(nn.Module):
    def __init__(self, scale_factor: int):
        super().__init__()
        self.scale_factor = scale_factor

    def forward(self, x):
        """ Upsampling along frequency-axis + Downsampling along time-axis """

        batch_size, in_channels, width = x.size()
        out_channels = in_channels * self.scale_factor
        width = width // self.scale_factor

        x = x.contiguous().view(batch_size, in_channels, width, self.scale_factor)
        x = x.permute(0, 1, 3, 2).contiguous()
        x = x.view(batch_size, out_channels, width)
        return x
        


class ConvLayerNorm(nn.Module):
    def __init__(self, d_hid, d_cond=None):
        super().__init__()

        if d_cond is None:
            d_cond = 128

        self.linear = nn.Linear(d_cond, 2 * d_hid, bias=False)

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
        
def layer_norm(x, dim=1, eps: float = 1e-14):
    mean = torch.mean(x, dim=dim, keepdim=True)
    var = torch.square(x - mean).mean(dim=dim, keepdim=True)
    return (x - mean) / torch.sqrt(var + eps)



class QuantizeContents(nn.Module):
    # https://github.com/zalandoresearch/pytorch-vq-vae/blob/master/vq-vae.ipynb
    def __init__(self, config, commitment_cost=0.25, decay=0.99, eps=1e-14):
        super().__init__()

        
        n_mel = config['Preprocess']['n_mel']
        n_embed = config['Model']['Reconstruction']['CodeSize']

        self._commitment_cost = commitment_cost
        self.decay = decay
        self.eps = eps
        self.embedding = nn.Embedding(n_embed, n_mel)

    def forward(self, cnt_emb):
        embed = (self.embedding.weight.detach()).transpose(0,1)
        embed = (embed)/(torch.norm(embed,dim=0))
        #cnt_emb = cnt_emb / torch.norm(cnt_emb, dim = 2, keepdim=True)
        flatten = cnt_emb.reshape(-1, cnt_emb.shape[-1]).detach()  # (B*T, C)
        
        dist = (
            flatten.pow(2).sum(1, keepdim=True)
            - 2 * flatten @ embed
            + embed.pow(2).sum(0, keepdim=True)
        )

        _, embed_ind = (-dist).max(1)                   # (B*T, C)
        embed_ind = embed_ind.view(*cnt_emb.shape[:-1]).detach().cpu().cuda()
        quantized = self.embedding(embed_ind)

        e_latent_loss = F.mse_loss(quantized.detach(), cnt_emb)
        q_latent_loss = F.mse_loss(quantized, cnt_emb.detach())
        loss = q_latent_loss + self._commitment_cost * e_latent_loss

        quantized = cnt_emb + (quantized - cnt_emb).detach()
        
        return quantized, loss

def instance_norm(x, eps: float = 1e-14):
    return layer_norm(x, dim=2)


