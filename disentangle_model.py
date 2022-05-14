import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.utils import spectral_norm

import numpy as np

from module import *
from recon_utils import PositionalEncoding


class DisentangleReconstruction(nn.Module):
    def __init__(self, config):
        super().__init__()
        device = 'cuda' if torch.cuda.is_available() else 'cpu'


        """ Configuration """
        # Structure
        d_enc = config['Model']['Reconstruction']['d_hid_encoder']
        d_spk = config['Model']['Reconstruction']['d_speaker']

        # number of speaker
        n_spk = config['Dataset']['num_speaker']


        """ Architecture """ 
        # 1. Positional Encoding & PreNet
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_enc))
        self.prenet = PreNet(config)

        # 2. Encoder, Decoder
        self.encoder = Encoder(config)
        self.decoder = Decoder(config)

        # 3. PostNet
        self.postnet = PostNet(config)

        # 4. Spk Classifier
        self.mlp_head = nn.Sequential(
            nn.Linear(d_spk, n_spk)
        )



    def forward(self, mel_input):
        batch_size = mel_input.shape[0]

        # 1) Pos & PreNet
        x = self.prenet(mel_input)

        x = torch.cat([self.cls_token.repeat(batch_size, 1, 1), x], dim=1)

        # 2) Encoder
        spk_emb, cnt_emb = self.encoder(x)

        # 3) Decoder & PostNet
        recon_mel = self.decoder(spk_emb, cnt_emb)
        post_mel = self.postnet(recon_mel)

    

        return recon_mel, post_mel, spk_emb, cnt_emb, self.mlp_head(spk_emb)



class DistributionClassifier(nn.Module):
    def __init__(self, config):
        """ Parameter """
        super().__init__()
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        d_spk = config['Model']['Reconstruction']['d_speaker']
        d_cnt = config['Model']['Reconstruction']['d_contents']

        n_layer = config['Model']['Classifier']['n_AttnBlock']


        # self.clssify_token = nn.Parameter(torch.zeros(1, d_spk)).to(device)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_cnt))
        self.pos_enc = PositionalEncoding(d_spk + d_cnt)

        f = spectral_norm
        self.Attn_Layers = nn.Sequential(*[MultiHeadAttention(config, d_spk + d_cnt) for _ in range(n_layer)])
        self.Feed_Layers = nn.Sequential(*[LinearFeedForward(config, d_spk + d_cnt) for _ in range(n_layer)])

        self.fc_net = nn.Sequential(
            f(nn.Linear(d_spk + d_cnt, 128)),
            nn.LeakyReLU(0.2, inplace=True),
            f(nn.Linear(128, 64)),
            nn.LeakyReLU(0.2, inplace=True),
            f(nn.Linear(64, 32)),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.last_linear = f(nn.Linear(32, 1))
        self.sigmoid = nn.Sigmoid()

    def forward(self, spk_emb, cnt_emb):
        """
            ? INPUT
            - spk_emb: (B, C)
            - cnt_emb: (B, T, C)
        """
        
        batch_size, timeSteps, _ = cnt_emb.shape

        hid = torch.cat([self.cls_token.repeat(batch_size, 1, 1), cnt_emb], dim=1)
        hid = torch.cat([spk_emb.unsqueeze(1).repeat(1, timeSteps + 1, 1), hid], dim=-1)

        hid = self.pos_enc(hid)

        for attn, fc in zip(self.Attn_Layers, self.Feed_Layers):
            hid = fc(attn(hid))

        out = self.last_linear(self.fc_net(hid[:, 0]))
        return self.sigmoid(out)





class MappingNetwork(nn.Module):
    def __init__(self, config):
        """
            FC * 8
        """
        super().__init__()
        n_layer = config['Model']['Classifier']['n_Mapping']
        d_spk = config['Model']['Reconstruction']['d_speaker']

        self.mapping = []
        
        f = spectral_norm
        for i in range(n_layer):
            self.mapping.append(f(nn.Linear(d_spk, d_spk)))
            self.mapping.append(nn.LayerNorm(d_spk))
            self.mapping.append(nn.LeakyReLU(0.2, inplace=True))
        
        self.mapping = nn.Sequential(*self.mapping)


    def forward(self, z):
        """
            ? INPUT
            - z: sampling noise     (B, d_spk)

            ? OUTPUT
            - output: mapped vector from this network   (B, d_spk)
        """
        return self.mapping(z)





# class MappingNetwork(nn.Module):
#     def __init__(self, config):
#         """
#             Normalize - FC * 8
#         """
#         super().__init__()
#         n_layer = config['Model']['Classifier']['n_Mapping']
#         d_spk = config['Model']['Reconstruction']['d_speaker']

#         self.normalize = PixelNormLayer(epsilon=1e-12)

#         self.mapping = []
#         for i in range(n_layer):
#             self.mapping.append(EqualLinear(d_spk, d_spk))
        
#         self.mapping = nn.Sequential(*self.mapping)


#     def forward(self, z):
#         """
#             ? INPUT
#             - z: sampling noise     (B, d_spk)

#             ? OUTPUT
#             - output: mapped vector from this network   (B, d_spk)
#         """
#         return self.mapping(self.normalize(z))
