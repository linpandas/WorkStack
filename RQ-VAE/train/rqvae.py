import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from interfaces import Stage1Model
from quantizations import RQBottleneck
from modules import Encoder, Decoder

class RQVAE(Stage1Model):
    def __init__(self,
                 *,
                 embed_dim=64,
                 n_embed=128,
                 decay=0.99,
                 loss_type='mse',
                 latent_loss_weight=0.25,
                 bottleneck_type='rq',
                 **kwargs):
        super().__init__()

        assert loss_type in ['mse', 'l1']

        self.encoder = Encoder(4096)
        self.decoder = Decoder(4096)

        if bottleneck_type == 'rq':
            self.quantizer = RQBottleneck(latent_dim=embed_dim,
                                          n_layer=3,
                                          n_embed=n_embed,
                                          decay=decay
                                          )
        else:
            raise ValueError("invalid 'bottleneck_type' (must be 'rq')")

        self.loss_type = loss_type
        self.latent_loss_weight = latent_loss_weight

    def forward(self, xs):
        z_e = self.encode(xs)
        z_q, quant_loss, code = self.quantizer(z_e)
        out = self.decode(z_q)
        return out, quant_loss, code

    def encode(self, x):
        z_e = self.encoder(x)
        return z_e

    def decode(self, z_q):
        out = self.decoder(z_q)
        return out

    @torch.no_grad()
    def get_codes(self, xs):
        z_e = self.encode(xs)
        _, _, code = self.quantizer(z_e)
        return code

    @torch.no_grad()
    def get_soft_codes(self, xs, temp=1.0, stochastic=False):
        assert hasattr(self.quantizer, 'get_soft_codes')

        z_e = self.encode(xs)
        soft_code, code = self.quantizer.get_soft_codes(z_e, temp=temp, stochastic=stochastic)
        return soft_code, code

    @torch.no_grad()
    def decode_code(self, code):
        z_q = self.quantizer.embed_code(code)
        decoded = self.decode(z_q)
        return decoded

    def compute_loss(self, out, quant_loss, code, xs=None, valid=False):

        if self.loss_type == 'mse':
            loss_recon = F.mse_loss(out, xs, reduction='mean')
        elif self.loss_type == 'l1':
            loss_recon = F.l1_loss(out, xs, reduction='mean')
        else:
            raise ValueError('incompatible loss type')

        loss_latent = quant_loss

        loss_total = loss_recon + self.latent_loss_weight * loss_latent

        return {
            'loss_total': loss_total,
            'loss_recon': loss_recon,
            'loss_latent': loss_latent,
            'codes': [code]
        }