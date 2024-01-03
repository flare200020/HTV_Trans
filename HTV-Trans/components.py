import torch
import torch.nn as nn
import numpy as np
import copy
from torch.nn.parameter import Parameter
import math
import torch.nn.functional as F
from VAES.HTPGM import HTPGM


class LinearUnit(nn.Module):
    def __init__(self, in_features, out_features, nonlinearity=nn.LeakyReLU(0.2)):
        super(LinearUnit, self).__init__()
        self.model = nn.Sequential(
                     nn.Linear(in_features, out_features),nonlinearity)

    def forward(self, x):
        return self.model(x)

class LossFunctions:
    eps = 1e-8

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
        if self.eps > 0.0:
            var = var + self.eps
        # print(x.shape,mu.shape)
        return -0.5 * torch.sum(np.log(2.0 * np.pi) + torch.log(var) + torch.pow(x - mu, 2) / var)


# Combine Network
class CombineNet(nn.Module):
    '''
    the fusion of z and stationarized x
    '''
    def __init__(self, z_dim, x_dim, h_dim, embd_h, layer_xz, z_layers, n_head, vocab_len, dropout, q_len, win_len,
                 device=torch.device('cuda:0')):
        super(CombineNet, self).__init__()
        self.z_dim = z_dim
        self.x_dim = x_dim
        self.h_dim = h_dim
        self.embd_h = embd_h
        self.layer_xz = layer_xz
        self.z_layers = z_layers
        self.n_head = n_head
        self.vocab_len = vocab_len
        self.dropout = dropout
        self.q_len = q_len
        self.win_len = win_len
        self.device = device
        # change HTPGM layer number here
        self.tvae = HTPGM(layers=3, input_shape=(3, win_len, self.x_dim), latent_dim=h_dim,
                         h_shape=(1, win_len, self.h_dim), output_shape=(1, win_len, self.x_dim)).to(device)
        '''layers of z to transformer'''
        self.transz_layers = nn.ModuleList([VariationalTransformer(n_time_series=h_dim, n_head=n_head, num_layer=layer_xz,
                                                  n_embd=h_dim, vocab_len=vocab_len, dropout=dropout, q_len=q_len,
                                                  win_len=win_len, scale_att=False, use_gcn=False, device=device) for l in range(self.z_layers)])
        self.trans_z = nn.Sequential(LinearUnit(x_dim, h_dim), LinearUnit(h_dim, h_dim))
        self.proj_h = nn.Sequential(LinearUnit(((h_dim) * 2) * self.z_layers, h_dim), LinearUnit(h_dim, h_dim))
        self.softplus = nn.Softplus()
        self.layernormz = nn.LayerNorm([win_len, h_dim])

    def infer_qz(self, x, x_embd, x_raw, x_raw_embd):
        z_list, latent_post_list = self.tvae.Inference(x_raw_embd)
        z_cat = z_list[0]
        for num in range(1, len(z_list)):
            z_cat += F.interpolate(z_list[num], scale_factor=(self.win_len / z_list[num].shape[-1]), mode='nearest')
        z_cat = z_cat.permute(0, 2, 1)
        for i, Transz in enumerate(self.transz_layers):
            '''more layers'''
            if(i==0):
                z_cat = self.layernormz(z_cat)
                z_forward = x_embd+z_cat
                h = Transz(z_forward)
                ztmp = h
            else:
                z_forward = z_cat+x_embd
                h = Transz(z_forward)
                ztmp = torch.cat((h, ztmp), dim=2)

        z_final = ztmp
        h = self.proj_h(z_final)
        x_given_z_mean, x_given_z_var, latent_prior_list = self.tvae.Generate(x_raw, z_list, h)

        return z_list, latent_post_list, h, x_given_z_mean, x_given_z_var, latent_prior_list

    def forward(self, x, x_embd, x_raw, x_raw_embd):
        x = x.float().squeeze(2).squeeze(-1)
        z_list, latent_post_list, h, x_given_z_mean, x_given_z_var, latent_prior_list = \
            self.infer_qz(x, x_embd, x_raw, x_raw_embd)
        return z_list, latent_post_list, h, x_given_z_mean, x_given_z_var, latent_prior_list


class Attention(nn.Module):
    def __init__(self, n_head, n_embd, win_len, scale, q_len, attn_pdrop=0.1, resid_pdrop=0.1, use_gcn=False):
        super(Attention, self).__init__()
        mask = torch.tril(torch.ones(win_len, win_len)).view(1, 1, win_len, win_len)
        self.register_buffer('mask_tri', mask)
        self.use_gcn = use_gcn
        self.n_head = n_head
        self.split_size = n_embd * self.n_head
        self.scale = scale
        self.q_len = q_len
        self.query_key = nn.Conv1d(n_embd, n_embd * n_head * 2, self.q_len)
        if use_gcn:
            self.value = Conv1D(n_embd * n_head, 0, n_embd)
            self.c_proj = Pooling(n_head)
        else:
            self.value = Conv1D(n_embd * n_head, 1, n_embd)
            self.c_proj = Conv1D(n_embd, 1, n_embd * self.n_head)
        self.attn_dropout = nn.Dropout(attn_pdrop)
        self.resid_dropout = nn.Dropout(resid_pdrop)

    def attn(self, query: torch.Tensor, key, value: torch.Tensor):
        activation = torch.nn.Softmax(dim=-1)
        pre_att = torch.matmul(query, key)
        if self.scale:
            pre_att = pre_att / math.sqrt(value.size(-1))
        mask = self.mask_tri[:, :, :pre_att.size(-2), :pre_att.size(-1)]
        pre_att = pre_att * mask + -1e9 * (1 - mask)
        pre_att = activation(pre_att)
        pre_att = self.attn_dropout(pre_att)
        attn = torch.matmul(pre_att, value)

        return attn

    def merge_heads(self, x):
        x = x.permute(0, 2, 1, 3).contiguous()
        new_x_shape = x.size()[:-2] + (x.size(-2) * x.size(-1),)
        return x.view(*new_x_shape)

    def split_heads(self, x, k=False):
        new_x_shape = x.size()[:-1] + (self.n_head, x.size(-1) // self.n_head)
        x = x.view(*new_x_shape)
        if k:
            return x.permute(0, 2, 3, 1)
        else:
            return x.permute(0, 2, 1, 3)

    def forward(self, x):
        value = self.value(x)
        qk_x = nn.functional.pad(x.permute(0, 2, 1), pad=(self.q_len - 1, 0))
        query_key = self.query_key(qk_x).permute(0, 2, 1)
        query, key = query_key.split(self.split_size, dim=2)
        query = self.split_heads(query)
        key = self.split_heads(key, k=True)
        value = self.split_heads(value)
        attn = self.attn(query, key, value)
        if not self.use_gcn:
            attn = self.merge_heads(attn)
        attn = self.c_proj(attn)
        attn = self.resid_dropout(attn)
        return attn

class Conv1D(nn.Module):
    def __init__(self, out_dim, rf, in_dim):
        super(Conv1D, self).__init__()
        self.rf = rf
        self.out_dim = out_dim
        self.in_dim = in_dim
        if rf == 1:
            w = torch.empty(in_dim, out_dim)
            nn.init.normal_(w, std=0.02)
            self.w = Parameter(w)
            self.b = Parameter(torch.zeros(out_dim))

    def forward(self, x):
        if self.rf == 1:
            size_out = x.size()[:-1] + (self.out_dim,)
            x = torch.addmm(self.b, x.view(-1, x.size(-1)), self.w)

            x = x.view(*size_out)
        else:
            size_out = x.size()[:-1] + (self.out_dim,)
            x = x.unsqueeze(-2).repeat(1, 1, self.out_dim // self.in_dim, 1)
            x = x.view(*size_out)
        return x

class Pooling(nn.Module):
    def __init__(self, n_head):
        super(Pooling, self).__init__()
        self.pool = nn.AvgPool1d(kernel_size=n_head)

    def forward(self, x):
        B, H, T, P = x.size()
        x = x.view(B, H, -1).permute(0, 2, 1)
        x = self.pool(x).squeeze().view(B, T, P)
        return x


class LayerNorm(nn.Module):
    "Construct a layernorm module in the OpenAI style (epsilon inside the square root)."

    def __init__(self, n_embd, e=1e-5):
        super(LayerNorm, self).__init__()
        self.g = nn.Parameter(torch.ones(n_embd))
        self.b = nn.Parameter(torch.zeros(n_embd))
        self.e = e

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = (x - mu).pow(2).mean(-1, keepdim=True)
        x = (x - mu) / torch.sqrt(sigma + self.e)
        return self.g * x + self.b


class MLP(nn.Module):
    def __init__(self, n_state, n_embd):
        super(MLP, self).__init__()
        n_embd = n_embd
        self.c_fc = Conv1D(n_state, 1, n_embd)
        self.c_proj = Conv1D(n_embd, 1, n_state)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        hidden1 = self.act(self.c_fc(x))
        hidden2 = self.c_proj(hidden1)
        return self.dropout(hidden2)


class Block(nn.Module):
    def __init__(self, n_head, win_len, n_embd, scale, q_len, use_gcn):
        super(Block, self).__init__()
        self.use_gcn = use_gcn
        self.attn = Attention(n_head, n_embd, win_len, scale, q_len, use_gcn=use_gcn)
        self.ln_1 = LayerNorm(n_embd)
        self.mlp = MLP(4 * n_embd, n_embd)
        self.ln_2 = LayerNorm(n_embd)
        if use_gcn:
            # For graph
            self.linear_map = nn.Linear(n_embd, n_embd)

    def forward(self, x, y):

        attn = self.attn(x)

        ln1 = self.ln_1(x + attn)
        if self.use_gcn:
            mlp = torch.mm(y, y.t()).t()
            mlp = self.linear_map(torch.matmul(ln1, mlp))
        else:
            mlp = self.mlp(ln1)

        hidden = self.ln_2(ln1 + mlp)
        return hidden


class SelfDefinedTransformer(nn.Module):
    def __init__(self, n_time_series, n_head, num_layer, n_embd, vocab_len, win_len, dropout, scale_att,
                 q_len, use_gcn, device=torch.device('cpu')):
        super(SelfDefinedTransformer, self).__init__()
        self.input_dim = n_time_series
        self.n_head = n_head
        self.num_layer = num_layer
        self.n_embd = n_embd
        self.vocab_len = vocab_len
        self.win_len = win_len
        self.dropout = dropout
        self.scale_att = scale_att
        self.q_len = q_len
        self.use_gcn = use_gcn
        self.device = device
        # The following is the implementation of this paragraph
        """ For positional encoding in Transformer, we use learnable position embedding.
        For covariates, following [3], we use all or part of year, month, day-of-the-week,
        hour-of-the-day, minute-of-the-hour, age and time-series-ID according to the granularities of datasets.
        age is the distance to the first observation in that time series [3]. Each of them except time series
        ID has only one dimension and is normalized to have zero mean and unit variance (if applicable).
        """
        if use_gcn:
            # If use gcn, position embedding is added with the input series.
            # Otherwise, position embedding is concatenated with the input series
            assert n_time_series == n_embd
            self.po_embed = nn.Embedding(vocab_len, n_time_series)
            block = Block(n_head, vocab_len, n_time_series, scale=scale_att, q_len=q_len, use_gcn=use_gcn)
        else:
            self.po_embed = nn.Embedding(vocab_len, n_embd)
            block = Block(n_head, vocab_len, n_time_series + n_embd, scale=scale_att, q_len=q_len, use_gcn=use_gcn)
        self.blocks = nn.ModuleList([copy.deepcopy(block) for _ in range(num_layer)])
        nn.init.normal_(self.po_embed.weight, std=0.02)

    def forward(self, x, y):
        batch_size = x.size(0)
        length = x.size(1)

        if self.use_gcn:
            embedding_sum = torch.zeros(batch_size, length, self.input_dim).to(self.device)
        else:
            embedding_sum = torch.zeros(batch_size, length, self.n_embd).to(self.device) #change 540
        position = torch.tensor(torch.arange(length), dtype=torch.long).to(self.device)
        po_embedding = self.po_embed(position)

        embedding_sum[:] = po_embedding
        if self.use_gcn:
            x = x + embedding_sum

        else:
            x = torch.cat((x, embedding_sum), dim=2)
            pass
        for block in self.blocks:

            x = block(x, y)
        return x


class VariationalTransformer(nn.Module):
    def __init__(self, n_time_series: int, n_head: int, num_layer: int, n_embd: int, vocab_len: int, dropout: float,
                 q_len: int, win_len: int, scale_att: bool = False, use_gcn: bool = False, device=torch.device('cpu')):
        """
        Args:
            n_time_series: Number of time series present in input
            n_head: Number of heads in the MultiHeadAttention mechanism
            num_layer: The number of transformer blocks in the model.
            n_embd: The dimention of Position embedding and time series ID embedding
            vocab_len: The number of vocabulary length
            dropout: The dropout for the embedding of the model.
            q_len:
            win_len:
            scale_att: whether use scale when calculating self-attention mask
            use_gcn: whether use gcn in transformer
        """
        super(VariationalTransformer, self).__init__()
        self.transformer = SelfDefinedTransformer(n_time_series, n_head, num_layer, n_embd, vocab_len, win_len,
                                                  dropout, scale_att, q_len, use_gcn, device)
        self._initialize_weights()
        self.n_embd = n_embd
        self.po_embed = nn.Embedding(vocab_len, n_embd)
        nn.init.normal_(self.po_embed.weight, std=0.02)
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor, y: torch.Tensor = None):
        """
        Args:
            x: Tensor of dimension (batch_size, seq_len, number_of_time_series)
            series_id: Optional id of the series in the dataframe. Currently  not supported
        Returns:

        """
        h = self.transformer(x, y)
        return h
