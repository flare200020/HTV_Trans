import torch
from torch import nn
import numpy as np
from torch.nn import functional as F

class HTPGM(nn.Module):
    '''
    HTPGM module
    '''
    def __init__(self, input_shape=(1, 96, 7), h_shape=(1, 96, 8), output_shape=(1, 96, 7), layers=3, latent_dim=512,
                 window_size=None):
        super(HTPGM, self).__init__()
        if window_size is None:
            window_size = [1, 1, 2, 4, 8, 16, 32]
        self.in_dim = input_shape[-2]
        self.layers = layers
        self.out_dim = input_shape[-2]
        self.target_dim = output_shape[-1]
        self.z_dim = [int(self.in_dim/i) for i in window_size]  # z_dim[0] is leave for the win_len
        self.latent_dim = latent_dim
        self.h_dim = h_shape[-1]
        self.win_len = input_shape[-2]
        self.beta = 0
        self.if_reshape = False
        # initialize required MLPs
        self.Linear_x = nn.Sequential(
            nn.Linear(self.in_dim, self.latent_dim * 2),
            nn.Linear(self.latent_dim * 2, self.latent_dim)
        )
        self.MLPz = nn.Sequential(
            nn.Linear(self.in_dim, round(self.latent_dim * 2)),
            nn.ReLU(),
            nn.Linear(round(self.latent_dim * 2), self.in_dim),
            nn.LayerNorm([self.in_dim]),
            nn.ReLU()
        )
        self.MLPR = nn.Sequential(
            nn.Linear(self.in_dim, round(self.latent_dim)),
            nn.ReLU(),
            nn.Linear(round(self.in_dim), self.out_dim),
            nn.Sigmoid()
        )
        self.Linear_mean = nn.Sequential(
            nn.Linear(self.h_dim, self.target_dim),
        )
        self.Linear_var = nn.Sequential(
            nn.Linear(self.h_dim, self.target_dim),
        )
        self.softplus = nn.Softplus()

        self.inference_layers = nn.ModuleList(inference_Layer(z_dim=self.z_dim[i+1], latent_dim=self.latent_dim, in_dim=self.in_dim, out_dim=self.in_dim) for i in range(self.layers))
        self.generate_layers = nn.ModuleList([generate_Layer(z_dim=self.z_dim[i], latent_dim=self.latent_dim, h_dim=self.h_dim, in_dim=self.z_dim[i+1], hin_dim=self.out_dim) for i in range(self.layers)])
        self.Proj_h = nn.Sequential(nn.Linear(self.h_dim, latent_dim*2), nn.Linear(latent_dim*2, self.latent_dim))
        self.Prior_z_mean = nn.Sequential(nn.Linear(self.win_len, self.win_len*2), nn.Linear(self.win_len*2, self.z_dim[self.layers]))
        self.Prior_z_var = nn.Sequential(nn.Linear(self.win_len, self.win_len * 2), nn.Linear(self.win_len * 2,  self.z_dim[self.layers]))
        self.layernorm = nn.LayerNorm([self.out_dim, self.latent_dim])

    def Inference(self, x):
        '''bottom up'''
        d = x.permute(0, 2, 1)
        latent_post_list = [None] * self.layers
        inference_layers = self.inference_layers
        z_list = [None] * (self.layers)

        for i in range(self.layers):
            if (i == self.layers - 1):
                d, z_post_mean, z_post_var = inference_layers[i](d)
                z = self.rt_gaussian(z_post_mean, z_post_var)
                z_list[i] = z

                latent_post_list[i] = (z_post_mean, z_post_var)
            else:
                d, z_post_mean, z_post_var = inference_layers[i](d)
                z = self.rt_gaussian(z_post_mean, z_post_var)
                z_list[i] = z
                latent_post_list[i] = (z_post_mean, z_post_var)
        return z_list, latent_post_list

    def Generate(self, x, z_list, h):
        '''top_down'''
        means = x.mean(1, keepdim=True).detach()
        stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5)
        latent_prior_list = [None] * (self.layers)
        h = self.Proj_h(h)
        h = self.layernorm(h)
        z_mean, z_var = self.Pz_prior(h)
        self.z_prior_mean = z_mean
        self.z_prior_var = (torch.ones(z_list[-1].size(), dtype=torch.float)).to(h.device)
        latent_prior_list[-1] = (self.z_prior_mean, self.z_prior_var)
        generate_layers = self.generate_layers
        for i in range(self.layers):
            id = self.layers - i - 1
            if id == 0:
                z = self.MLPz(z_list[id])
                h_temp = h.permute(0, 2, 1)
                x_given_z_mean = self.Linear_mean((z + h_temp).permute(0,2,1))
                x_given_z_var = self.softplus(self.Linear_var((z + h_temp).permute(0,2,1)))

            else:
                z_prior_mean, z_prior_var, h = generate_layers[id](z_list[id], h)
                latent_prior_list[id - 1] = (z_prior_mean, z_prior_var)

        x_given_z_mean = x_given_z_mean * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.win_len, 1))
        x_given_z_mean = x_given_z_mean + (means[:, 0, :].unsqueeze(1).repeat(1, self.win_len, 1))
        return x_given_z_mean, x_given_z_var, latent_prior_list

    def Pz_prior(self, h):
        """
        Top layer prior distribution
        """
        mean = self.Prior_z_mean(h.permute(0,2,1))
        var = self.softplus(self.Prior_z_var(h.permute(0,2,1)))
        return mean, var

    def sample_generation(self, z, start_from, h):
        generate_layers = self.generate_layers
        z_i = z
        h_temp = self.Proj_h(h)
        h_temp = self.layernorm(h_temp)
        for i in range(start_from):
            id = start_from - i - 1
            if id == 0:
                z = self.MLPz(z_i)
                x_given_z_mean = self.Linear_mean((z + h_temp.permute(0, 2, 1)).permute(0,2,1))
                x_given_z_var = self.softplus(self.Linear_var((z + h_temp.permute(0, 2, 1)).permute(0,2,1)))
                recon_x = self.rt_gaussian(x_given_z_mean, x_given_z_var)
            else:
                z_prior_mean, z_prior_var, h = generate_layers[id](z_i, h_temp)
                z_i = self.rt_gaussian(z_prior_mean, z_prior_var)

        return recon_x

    def loss_LLH(self, x, x_mu, x_var):
        loglikelihood = self.log_normal(x.float(), x_mu.float(), x_var.float())
        return loglikelihood

    def loss_KL(self, z_mean_posterior_forward, z_var_posterior_forward, z_mean_prior_forward,
                z_var_prior_forward):
        kld_z_forward = 0.5 * torch.sum(torch.log(z_var_prior_forward) - torch.log(z_var_posterior_forward) +
                                        ((z_var_posterior_forward + torch.pow(
                                            z_mean_posterior_forward - z_mean_prior_forward, 2)) /
                                         z_var_prior_forward) - 1)

        return kld_z_forward

    def loss_F(self, x, x_given_z_mean, x_given_z_var, latent_prior_list, latent_post_list):
        llh = self.loss_LLH(x, x_given_z_mean, x_given_z_var)
        loss = -llh + self.loss_KL(latent_post_list[-1][0], latent_post_list[-1][1], latent_prior_list[-1][0],
                                   latent_prior_list[-1][1])

        for i in range(len(latent_prior_list) - 1):
            loss += self.loss_KL(latent_post_list[i][0], latent_post_list[i][1], latent_prior_list[i][0],
                                 latent_prior_list[i][1])

        return llh, loss + llh

    # Code for sampling
    def rt_gaussian(self, mean, var, random_sampling=True):
        if random_sampling is True:
            eps = torch.randn_like(var)
            std = var ** 0.5
            z = mean + eps * std
            return z
        else:

            return mean

    def log_normal(self, x, mu, var):
        """Logarithm of normal distribution with mean=mu and variance=var
           log(x|μ, σ^2) = loss = -0.5 * Σ log(2π) + log(σ^2) + ((x - μ)/σ)^2

        Args:
           x: (array) corresponding array containing the input
           mu: (array) corresponding array containing the mean
           var: (array) corresponding array containing the variance

        Returns:
           output: (array/float) depending on average parameters the result will be the mean
                                of all the sample losses or an array with the losses per sample
        """

        eps = 1e-8
        if eps > 0.0:
            var = var + eps

        return -0.5 * torch.sum(np.log(2.0 * np.pi) + torch.log(var) + torch.pow(x - mu, 2) / var)

    def forward(self, data, h):
        z_list, latent_post_list = self.Inference(data)
        x_given_z_mean, x_given_z_var, latent_prior_list = self.Generate(data, z_list, h)
        llh, kl_loss = self.loss_F(data, x_given_z_mean, x_given_z_var, latent_prior_list, latent_post_list)
        return x_given_z_mean, x_given_z_var, llh, kl_loss


class inference_Layer(nn.Module):
    '''
    input d:  di
    output d:di+1  ,z_post_mean, z_post_var:the distribution of qfhi
    '''

    def __init__(self, z_dim=96, latent_dim=48, in_dim = 7, out_dim = 7):
        super(inference_Layer, self).__init__()
        self.latent_dim = latent_dim
        self.in_dim = in_dim  # for input dim
        self.z_dim = z_dim  # for z dim
        self.out_dim = out_dim  # for d dim
        self.layernorm = nn.LayerNorm([out_dim])

        self.MLP = nn.Sequential(
            nn.Linear(self.in_dim, self.latent_dim * 2),
            nn.ReLU(),
            nn.Linear(self.latent_dim*2, self.out_dim),
            nn.LayerNorm(self.out_dim),
            nn.ReLU()
        )
        self.Linear_mean_i = nn.Sequential(
            nn.Linear(self.out_dim, self.z_dim),
        )
        self.Linear_var_i = nn.Sequential(
            nn.Linear(self.out_dim, self.z_dim),
        )
        self.softplus = nn.Softplus()

    def forward(self, d):
        d = (self.MLP(d) + d)
        z_post_mean = self.Linear_mean_i(d)
        z_post_var = self.softplus(self.Linear_var_i(d))
        return d, z_post_mean, z_post_var


class generate_Layer(nn.Module):
    '''
    input d: di z_prior_mean, z_prior_var:the distribution of p_zi_given_zi+1
    output z_prior_mean, z_prior_var:the distribution of p_z_given_x  , z_prior_list: top k layer's distribution
    '''

    def __init__(self, z_dim=96, latent_dim=48, h_dim=48, in_dim=7, hin_dim=7):
        super(generate_Layer, self).__init__()
        self.latent_dim = latent_dim
        self.z_dim = z_dim
        self.in_dim = in_dim  # for input dim
        self.hin_dim = hin_dim  # for mlp d dim
        self.h_dim = h_dim  # for dynamic channel dim
        self.layernorm = nn.LayerNorm([in_dim])
        self.MLP = nn.Sequential(
            nn.Linear(self.in_dim, self.latent_dim * 2),
            nn.ReLU(),
            nn.Linear(self.latent_dim * 2, self.latent_dim * 2),
            nn.ReLU(),
            nn.Linear(self.latent_dim * 2, self.in_dim),
            nn.LayerNorm(self.in_dim),
            nn.ReLU()
        )
        self.Linear_mean_g = nn.Sequential(
            nn.Linear(self.hin_dim, self.z_dim)
        )
        self.Linear_var_g = nn.Sequential(
            nn.Linear(self.hin_dim, self.z_dim)
        )
        self.softplus = nn.Softplus()

    def forward(self, z, h):
        h_temp = h.permute(0, 2, 1)
        z = (self.MLP(z))
        z = F.interpolate(z, scale_factor=(h_temp.shape[-1]/z.shape[-1]), mode='nearest')
        z_prior_mean = self.Linear_mean_g(z+h_temp)
        z_prior_var = self.softplus(self.Linear_var_g(z+h_temp))
        return z_prior_mean, z_prior_var, h


