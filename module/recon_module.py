import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.utils import spectral_norm

from .recon_layer import *
from recon_utils import PositionalEncoding


class Encoder(nn.Module):
    def __init__(self, config):
        super().__init__()

        """ Parameter """
        d_enc = config['Model']['Reconstruction']['d_hid_encoder']
        d_spk = config['Model']['Reconstruction']['d_speaker']
        d_cnt = config['Model']['Reconstruction']['d_contents']
        self.down_rate = config['Model']['Reconstruction']['downSampling_rate']

        
        n_AttnBlock = config["Model"]['Reconstruction']['n_EncAttnBlock']
        n_ConvBlock = config["Model"]['Reconstruction']['n_EncConvBlock']

        """ Architecture """
        self.pos_enc = PositionalEncoding(d_enc)

        self.enc_attn = nn.Sequential(*[
            EncAttnBlock(config) for _ in range(n_AttnBlock)
        ])

        self.spk_net = SpeakerNetwork(config)

        self.cnt_net = nn.Sequential(
            *[EncConvBlock(config) for i in range(n_ConvBlock)],
            nn.Linear(d_enc, d_cnt),
            nn.LayerNorm(d_cnt),
            nn.Tanh()
        )

        #self.downsampling = nn.AvgPool1d(down_rate, stride=down_rate)   # AvgPool will make degrade.



    def forward(self, input):
        batch_size, _, channels = input.shape

        # Feed Attention Blocks
        input = self.pos_enc(input)
        hid = self.enc_attn(input)

        # Speaker Embedding
        spk_emb = hid[:, 0]                 # (B, d_enc)
        spk_emb = self.spk_net(spk_emb)     # (B, d_spk)

        # Content Embedding (+ Downsampling)
        cnt_emb = self.cnt_net(hid[:, 1:])[:, ::self.down_rate]  # (B, T_down, d_cnt)
        
        
        return spk_emb, cnt_emb # Speaker Embedding, Content Embedding



class Decoder(nn.Module):
    def __init__(self, config):
        super().__init__()

        """ Parameter """
        n_mel = config['Preprocess']['n_mel']

        d_spk = config['Model']['Reconstruction']['d_speaker']
        d_cnt = config['Model']['Reconstruction']['d_contents']
        d_dec = config['Model']['Reconstruction']['d_hid_decoder']
        dropout = config['Model']['Reconstruction']['dropout']
        self.down_rate = config['Model']['Reconstruction']['downSampling_rate']

        n_AttnBlock = config["Model"]['Reconstruction']['n_DecAttnBlock']
        n_ConvBlock = config["Model"]['Reconstruction']['n_DecConvBlock']
        

        """ Architecture """
        self.dec_attn = nn.Sequential(*[
            DecAttnBlock(config) for _ in range(n_AttnBlock)
        ])

        self.pre_linear = nn.Sequential(
            nn.Linear(d_cnt, d_dec),
            nn.Dropout(dropout),
            nn.LayerNorm(d_dec),
            nn.GELU()
        )
        self.pos_dec = PositionalEncoding(d_dec)

        self.post_linear = nn.Linear(d_dec, n_mel)

        self.dense_net = nn.Sequential(*[
            nn.Sequential(nn.Linear(d_spk, d_spk), nn.ReLU() if i != 8 else nn.Tanh()) for i in range(8)
        ])

        self.conv_net = nn.Sequential(*[
            DecConvBlock(config) for _ in range(n_ConvBlock)
        ])

        #self.upsampling = nn.ConvTranspose1d(d_spk + d_cnt, d_dec, kernel_size=down_rate, stride=down_rate)



    def forward(self, spk_emb, cnt_emb):
        """
        ? INPUT
        - spk_emb: (B, C) or (B, 1, C)
        - cnt_emb: (B, T, C)

        ? OUTPUT
        - output: (B, T, C)
        """
        assert len(spk_emb.shape) == 2 or len(spk_emb.shape) == 3, "[recon_module.py]"

        nb_frames = cnt_emb.shape[1]
        if len(spk_emb.shape) == 2:
            spk_emb = spk_emb.unsqueeze(1)


        # upsampling
        hid = self.pre_linear(cnt_emb).contiguous().transpose(1, 2)
        hid = F.interpolate(hid, scale_factor=self.down_rate, mode='nearest').contiguous().transpose(1, 2)
        hid = self.pos_dec(hid)
        # hid: (B, T, C)

        # condition embedding (for Conditional LayerNorm)
        cond = self.dense_net(spk_emb)

        # Feed Conv layers
        res = hid
        for layer in self.conv_net:
            hid = layer(hid, cond=cond)
        hid = hid + res
        
        # Feed Attention Blocks
        for attn in self.dec_attn:
            hid = attn(hid, cond=cond)  # (B, T, C)


        return self.post_linear(hid)



class SpeakerNetwork(nn.Module):
    def __init__(self, config):
        super().__init__()

        d_enc = config["Model"]["Reconstruction"]["d_hid_encoder"]
        d_spk = config["Model"]["Reconstruction"]["d_speaker"]

        n_spk = config["Model"]["Reconstruction"]["n_speaker"]

        self.fc_net = []

        for i in range(n_spk):
            self.fc_net.append(nn.Linear(d_spk if i != 0 else d_enc, d_spk))
            self.fc_net.append(nn.ReLU())

        self.fc_net = nn.Sequential(*self.fc_net)

    def forward(self, x):
        out = self.fc_net(x)
        return out.div(torch.norm(out, p=2, dim=-1, keepdim=True))




class PreNet(nn.Module):
    def __init__(self, config):
        """
        1) Conv1d + BatchNorm1d + dropout   \\ (n_mel -> d_hid)
        2) Conv1d + BatchNorm1d + dropout   \\ (n_mel -> d_hid)
        3) linear                           \\ (d_hid -> d_hid)
        """
        
        super().__init__()
        n_mel = config["Preprocess"]["n_mel"]
        d_hid = config["Model"]["Reconstruction"]["d_hid_encoder"]
        dropout = config["Model"]["Reconstruction"]["dropout"]

        self.prenet = nn.Sequential(
            nn.Linear(n_mel, d_hid, bias=False),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_hid, d_hid, bias=False),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        self.linear = nn.Linear(d_hid, d_hid)

    def forward(self, x):
        x = self.prenet(x)
        return self.linear(x)





class PostNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        n_mel = config["Preprocess"]["n_mel"]

        self.conv = nn.ModuleList()

        self.conv.append(
            nn.Sequential(
                nn.Conv1d(n_mel, 512, kernel_size=5, stride=1, padding=2, dilation=1),
                nn.BatchNorm1d(512))
        )

        for i in range(1, 5 - 1):
            self.conv.append(
                nn.Sequential(
                    nn.Conv1d(512, 512, kernel_size=5, stride=1, padding=2, dilation=1),
                    nn.BatchNorm1d(512))
            )

        self.conv.append(
            nn.Sequential(
                nn.Conv1d(512, n_mel, kernel_size=5, stride=1, padding=2, dilation=1),
                nn.BatchNorm1d(n_mel))
        )

    def forward(self, x):
        out = x.contiguous().transpose(1, 2)
        for i in range(len(self.conv) - 1):
            out = torch.tanh(self.conv[i](out))

        out = self.conv[-1](out).contiguous().transpose(1, 2)

        # Residual Connection
        # ! Comment: 의외라 깜짝 놀랐는데 이 한 줄이 loss 입장에서 상당히 중요함.
        out = out + x

        return out


if __name__ == "__main__":
    import yaml
    config = yaml.load(
        open("./config.yaml", "r"), Loader=yaml.FullLoader
    )

    input = torch.zeros(8, 128, 80)

    model = DisentangleReconstruction(config)

    from torchsummaryX import summary
    summary(model, input)


