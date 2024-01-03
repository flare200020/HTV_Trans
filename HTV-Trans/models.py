import torch
import torch.nn as nn
import torch.nn.functional as F
from components import LossFunctions, CombineNet
from utils import LinearUnit, DataEmbedding
from layers.Embed import DataEmbedding
from RevIN import RevIN


class StackedVAGT(nn.Module):
    def __init__(self, layer_xz=1, z_layers=1, n_head=8, x_dim=36, z_dim=15, h_dim=20, embd_h=256, embd_s=256, q_len=1,
                 vocab_len=128, win_len=20, horizon=12, label_len=24, dropout=0.1, beta=0.1, anneal_rate=1, max_beta=1, device=torch.device('cuda:0'), decoder_type = 'fcn'):
        super(StackedVAGT, self).__init__()
        """
        In this class we will merge inference and generation altogether for simplification!
        """
        self.beta = beta
        self.max_beta = max_beta
        self.anneal_rate = anneal_rate
        self.layer_xz = layer_xz
        self.z_layers = z_layers
        self.n_head = n_head
        self.x_dim = x_dim
        self.z_dim = z_dim
        self.h_dim = h_dim
        self.embd_h = embd_h
        self.embd_s = embd_s
        self.q_len = q_len
        self.vocab_len = vocab_len
        self.win_len = win_len
        self.horizon = horizon
        self.label_len = label_len
        self.dropout = dropout
        self.d_layers = 1
        self.device = device
        self.losses = LossFunctions()

        # For Inference and Generation
        self.revin = RevIN(x_dim)
        self.inference = CombineNet(z_dim, x_dim, h_dim, embd_h, layer_xz, z_layers, n_head,
                                      vocab_len, dropout, q_len, win_len, device)
        self.end_conv = nn.Conv2d(win_len,horizon, kernel_size=(1,1), bias=True)
        self.end_fc = nn.Linear(h_dim, x_dim)
        self.decoder_type = decoder_type
        self.htox = LinearUnit(h_dim, x_dim)

        self.decoder_layer = nn.TransformerDecoderLayer(d_model=h_dim, nhead=n_head, batch_first=True)  # d_model = 768, nhead= 12---the number of heads in the multiheadattention models
        self.trans_decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=layer_xz)
        self.enc_embedding = DataEmbedding(x_dim, h_dim, 'timeF', 'h', dropout)
        self.dec_embedding = DataEmbedding(x_dim, h_dim, 'timeF', 'h', dropout)
        self.encx = nn.Sequential(LinearUnit(x_dim, x_dim*2), LinearUnit(x_dim*2, x_dim))
        self.layernorm_x = nn.LayerNorm([win_len, x_dim])

    def loss_LLH(self, x, x_mu, x_logsigma):
        loglikelihood = self.losses.log_normal(x.float(), x_mu.float(), x_logsigma)
        return loglikelihood

    def loss_KL(self, z_mean_posterior_forward, z_logvar_posterior_forward, z_mean_prior_forward,
                z_logvar_prior_forward):
        z_var_posterior_forward = torch.exp(z_logvar_posterior_forward)
        z_var_prior_forward = torch.exp(z_logvar_prior_forward)

        kld_z_forward = 0.5 * torch.sum(torch.log(z_var_prior_forward) - torch.log(z_var_posterior_forward) +
                                        ((z_var_posterior_forward + torch.pow(
                                            z_mean_posterior_forward - z_mean_prior_forward, 2)) /
                                         z_var_prior_forward) - 1)
        return kld_z_forward

    def mse_loss(self, x, target):
        loss = torch.nn.MSELoss(reduction='mean')
        output = loss(x, target)

        return output

    def mae_loss(self, x, target):
        loss = torch.nn.L1Loss(reduction='mean')
        output = loss(x, target)
        return output

    def forward (self, x, x_mark_enc, x_dec, x_mark_dec, enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        x_raw = x.float()
        x_norm = self.revin(x_raw, 'norm')
        x_norm_embed = self.enc_embedding(x_norm, x_mark_enc)
        x_raw_embed = self.enc_embedding(x_raw, x_mark_enc)

        z_list, latent_post_list, h_out, x_mu, x_var, latent_prior_list = self.inference(x_norm, x_norm_embed, x_raw, x_raw_embed)
        # reconstruct x_raw
        llh, kl_loss = self.inference.tvae.loss_F(x_raw, x_mu, x_var, latent_prior_list, latent_post_list)

        #  fc based predictor
        h_out = h_out.unsqueeze(1)
        h_out = h_out.permute(0, 2, 1, 3)
        output = self.end_conv((h_out))  # B, T*C, N, 1

        output = self.end_fc(output)
        output = output.permute(0, 1, 3, 2)
        output = output.squeeze(-1)
        output = self.revin(output, 'denorm')
        output = output.unsqueeze(-1)

        return x_mu, x_var, llh, kl_loss, output
