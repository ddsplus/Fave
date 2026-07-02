import torch.nn as nn
import torch as th
import numpy as np
import math
import torch
import torch.nn.functional as F
from timm.models.vision_transformer import PatchEmbed, Attention, Mlp

from scipy import integrate
from scipy.stats import norm
from torch.nn.init import xavier_normal_, constant_, xavier_uniform_


class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()
        # dim must equal head_dim (hidden_size // num_heads)
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base

        inv_freq = 1.0 / (self.base ** (torch.arange(0, dim, 2).float().to(device) / dim))
        self.register_buffer("inv_freq", inv_freq)

        self._set_cos_sin_cache(seq_len=max_position_embeddings, device=device, dtype=torch.get_default_dtype())

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)

        freqs = torch.einsum("i,j->ij", t, self.inv_freq)

        emb = torch.cat((freqs, freqs), dim=-1)

        self.register_buffer("cos_cached", emb.cos()[None, None, :, :].to(device=device, dtype=dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :].to(device=device, dtype=dtype), persistent=False)

    def forward(self, x, seq_len=None):
        # x: [B, Heads, Seq_Len, Head_Dim]
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)

        return (
            self.cos_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
            self.sin_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
        )

def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin):
    """
    q, k: [Batch, Heads, Seq_Len, Head_Dim]
    cos, sin: [1, 1, Seq_Len, Head_Dim]
    """
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

class DropPath(nn.Module):
    """
    Drop paths (Stochastic Depth) per sample.
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x

        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()

        # Scale by 1/keep_prob to preserve expected value
        return x.div(keep_prob) * random_tensor


class SiLU(nn.Module):
    def forward(self, x):
        return x * th.sigmoid(x)

class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias


class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-8):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x):
        norm_x = x.pow(2).mean(-1, keepdim=True)
        x_normed = x * torch.rsqrt(norm_x + self.eps)
        return self.weight * x_normed


class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."

    def __init__(self, hidden_size, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(hidden_size, hidden_size * 4)
        self.w_2 = nn.Linear(hidden_size * 4, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.init_weights()

    def init_weights(self):
        nn.init.xavier_normal_(self.w_1.weight)
        nn.init.constant_(self.w_1.bias, 0)
        nn.init.xavier_normal_(self.w_2.weight)
        nn.init.constant_(self.w_2.bias, 0)

    def forward(self, hidden):
        hidden = self.w_1(hidden)
        activation = 0.5 * hidden * (1 + torch.tanh(math.sqrt(2 / math.pi) * (hidden + 0.044715 * torch.pow(hidden, 3))))
        return self.w_2(self.dropout(activation))

class MultiHeadedAttention(nn.Module):
    def __init__(self, heads, hidden_size, dropout):
        super().__init__()
        assert hidden_size % heads == 0
        self.size_head = hidden_size // heads
        self.num_heads = heads
        self.linear_layers = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for _ in range(3)])
        self.w_layer = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(p=dropout)
        self.init_weights()

        self.rope = RotaryEmbedding(self.size_head)

    def init_weights(self):
        nn.init.xavier_normal_(self.w_layer.weight)

    def forward(self, q, k, v, mask=None):
        batch_size = q.shape[0]
        seq_len = q.shape[1]
        q, k, v = [l(x).view(batch_size, -1, self.num_heads, self.size_head).transpose(1, 2) for l, x in zip(self.linear_layers, (q, k, v))]

        cos, sin = self.rope(q, seq_len)

        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        corr = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(q.size(-1))

        if mask is not None:
            mask = mask.unsqueeze(1).repeat([1, corr.shape[1], 1]).unsqueeze(-1).repeat([1,1,1,corr.shape[-1]])
            corr = corr.masked_fill(mask == 0, -1e9)
        prob_attn = F.softmax(corr, dim=-1)
        if self.dropout is not None:
            prob_attn = self.dropout(prob_attn)
        hidden = torch.matmul(prob_attn, v)
        hidden = self.w_layer(hidden.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.size_head))

        return hidden


class TransformerBlock(nn.Module):
    def __init__(self, hidden_size, attn_heads, dropout, drop_path=0.0, gamma_init=1e-2):
        super(TransformerBlock, self).__init__()

        # Self-attention
        self.norm1 = RMSNorm(hidden_size)
        self.attention = MultiHeadedAttention(heads=attn_heads, hidden_size=hidden_size, dropout=dropout)

        # Cross-attention
        self.norm2 = RMSNorm(hidden_size)
        self.cross_attention = MultiHeadedAttention(heads=attn_heads, hidden_size=hidden_size, dropout=dropout)

        # Feed-Forward
        self.norm3 = RMSNorm(hidden_size)
        self.feed_forward = PositionwiseFeedForward(hidden_size=hidden_size, dropout=dropout)

        # Gating
        self.se_reduction_dim = max(hidden_size // 4, 8)
        self.se_relu = nn.Linear(hidden_size, self.se_reduction_dim, bias=False)
        self.se_weight1 = nn.Linear(self.se_reduction_dim, hidden_size, bias=False)
        self.se_weight2 = nn.Linear(self.se_reduction_dim, hidden_size, bias=False)

        self.gamma = nn.Parameter(gamma_init * torch.ones(hidden_size), requires_grad=True)

        self.dropout = nn.Dropout(p=dropout)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()


    def ms_senet_gating(self, x_self, x_cross):
        """
        Dynamic fusion of self-attention and cross-attention outputs.
        Args:
            x_self: result from self-attention path
            x_cross: result from cross-attention path
        Returns:
            weighted fusion of the two inputs
        """
        sum_x = x_self + x_cross

        avg_x = sum_x[:, -1, :].unsqueeze(1)

        se_feat = F.relu(self.se_relu(avg_x))

        w1 = self.se_weight1(se_feat)
        w2 = self.se_weight2(se_feat)

        weights = F.softmax(torch.stack([w1, w2], dim=0), dim=0)

        return x_self * weights[0] + x_cross * weights[1]


    def forward(self, hidden, c, mask):
        residual = hidden
        norm_x = self.norm1(hidden)
        self_out = self.attention(norm_x, norm_x, norm_x, mask=mask)

        x_self = residual + self.drop_path(self.dropout(self_out))

        # Step 2: Cross-Attention & Gating
        if c is not None:
            norm_c = self.norm2(x_self)
            cross_out = self.cross_attention(norm_c, c, c, mask=None)
            cross_out = self.dropout(cross_out)

            fused_out = self.ms_senet_gating(x_self, cross_out)

            correction = fused_out - x_self
            hidden = x_self + self.drop_path(self.gamma * correction)
        else:
            hidden = x_self

        # Step 3: Feed Forward
        residual = hidden
        norm_x = self.norm3(hidden)
        ffn_out = self.feed_forward(norm_x)

        hidden = residual + self.drop_path(self.dropout(ffn_out))

        return hidden

class Transformer_rep(nn.Module):
    def __init__(self, args):
        super(Transformer_rep, self).__init__()
        self.hidden_size = args.hidden_size
        self.heads = 4
        self.dropout = args.dropout
        self.n_blocks = args.num_blocks
        self.last = args.last

        drop_path_rate = args.drop_path_rate
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, self.n_blocks)]

        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(
                hidden_size=self.hidden_size,
                attn_heads=self.heads,
                dropout=self.dropout,
                drop_path=dpr[i],
                gamma_init=args.gamma_init
             )
             for i in range(self.n_blocks)]
        )

    def forward(self, hidden, c, mask):
        i = 0
        encode = hidden

        for transformer in self.transformer_blocks:
            i += 1
            hidden = transformer(hidden, c, mask)

            if i == (self.n_blocks - self.last):
                encode = hidden

        return hidden, encode

class Fave_xstart(nn.Module):
    """Core denoising module."""
    def __init__(self, hidden_size, args):
        super(Fave_xstart, self).__init__()
        self.hidden_size = hidden_size
        self.linear_item = nn.Linear(self.hidden_size, self.hidden_size)
        self.linear_xt = nn.Linear(self.hidden_size, self.hidden_size)
        self.linear_t = nn.Linear(self.hidden_size, self.hidden_size)
        time_embed_dim = self.hidden_size * 4
        self.t_time_embed = nn.Sequential(nn.Linear(self.hidden_size, time_embed_dim), SiLU(), nn.Linear(time_embed_dim, self.hidden_size))
        self.r_time_embed = nn.Sequential(nn.Linear(self.hidden_size, time_embed_dim), SiLU(), nn.Linear(time_embed_dim, self.hidden_size))
        self.fuse_linear = nn.Linear(self.hidden_size*3, self.hidden_size)
        self.att = Transformer_rep(args)

        self.lambda_uncertainty = args.lambda_uncertainty
        self.dropout = nn.Dropout(args.dropout)
        self.norm_fm_rep = LayerNorm(self.hidden_size)

        self.item_num = args.item_num
        self.out_dims = [512, 2048]
        self.act_func = 'tanh'

        out_dims_temp = [self.hidden_size] + self.out_dims + [self.item_num]
        decoder_modules = []
        for d_in, d_out in zip(out_dims_temp[:-1], out_dims_temp[1:]):
            decoder_modules.append(nn.Linear(d_in, d_out))
            if self.act_func == 'relu':
                decoder_modules.append(nn.ReLU())
            elif self.act_func == 'sigmoid':
                decoder_modules.append(nn.Sigmoid())
            elif self.act_func == 'tanh':
                decoder_modules.append(nn.Tanh())
            elif self.act_func == 'leaky_relu':
                decoder_modules.append(nn.LeakyReLU())
            else:
                raise ValueError
        decoder_modules.pop()
        self.decoder = nn.Sequential(*decoder_modules)

        self.xavier_normal_initialization(self.decoder)


    def xavier_normal_initialization(self, module):
        r""" using `xavier_normal_`_ in PyTorch to initialize the parameters in
        nn.Embedding and nn.Linear layers. For bias in nn.Linear layers,
        using constant 0 to initialize.
        .. _`xavier_normal_`:
            https://pytorch.org/docs/stable/nn.init.html?highlight=xavier_normal_#torch.nn.init.xavier_normal_
        Examples:
            >>> self.apply(xavier_normal_initialization)
        """
        if isinstance(module, nn.Linear):
            xavier_normal_(module.weight.data)
            if module.bias is not None:
                constant_(module.bias.data, 0)


    def timestep_embedding(self, timesteps, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.

        :param timesteps: a 1-D Tensor of N indices, one per batch element.
                        These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an [N x dim] Tensor of positional embeddings.
        """
        half = dim // 2
        freqs = th.exp(-math.log(max_period) * th.arange(start=0, end=half, dtype=th.float32) / half).to(device=timesteps.device)
        args = timesteps[:, None].float() * freqs[None]
        embedding = th.cat([th.cos(args), th.sin(args)], dim=-1)
        if dim % 2:
            embedding = th.cat([embedding, th.zeros_like(embedding[:, :1])], dim=-1)
        return embedding


    def forward(self, rep_item, x_t, t, r, mask_seq, state="pretrain"):
        emb_t = self.t_time_embed(self.timestep_embedding(t, self.hidden_size))
        emb_r = self.r_time_embed(self.timestep_embedding(r-t, self.hidden_size))

        time_condition = emb_r + emb_t
        lambda_uncertainty = th.normal(mean=th.full(rep_item.shape, self.lambda_uncertainty), std=th.full(rep_item.shape, self.lambda_uncertainty)).to(x_t.device)

        rep_item_New = rep_item + (lambda_uncertainty * (x_t + time_condition).unsqueeze(1))

        condition_cross = rep_item
        if state == 'pretrain':
            condition_cross = None

        rep_fave, encode = self.att(rep_item_New, condition_cross, mask_seq)

        rep_fave = self.norm_fm_rep(self.dropout(rep_fave))

        out = rep_fave[:, -1, :]

        encoded = encode[:, -1, :]

        decode = self.decoder(encoded)

        return out, decode

class Fave(nn.Module):
    def __init__(self, args,):
        super(Fave, self).__init__()
        self.hidden_size = args.hidden_size
        self.xstart_model = Fave_xstart(self.hidden_size, args)
        self.eps = args.eps
        self.sample_N = args.sample_N
        self.eps_reverse = args.eps_reverse
        self.m_logNorm = args.m_logNorm
        self.s_logNorm = args.s_logNorm
        self.s_modsamp = args.s_modsamp
        self.sampling_method = args.sampling_method

    def from_flattened_numpy(self, x, shape):
        """Form a torch tensor with the given `shape` from a flattened numpy array `x`."""
        return torch.from_numpy(x.reshape(shape))

    def to_flattened_numpy(self, x):
        """Flatten a torch tensor `x` and convert it to numpy."""
        return x.detach().cpu().numpy().reshape((-1,))

    def euler_sampler(self, item_rep, mask_seq, z0, stage="pretrain"):
        with torch.no_grad():
            if stage == "pretrain":
                N = self.sample_N
            else:
                N = 1

            device = next(self.xstart_model.parameters()).device
            shape = item_rep[:,-1,:].shape

            x = z0.to(device)

            dt = 1./ N
            eps = self.eps_reverse          # default: 1e-3

            extra = (1 / self.eps_reverse) - 1

            for i in range(N):

                num_t = i / N * (self.T - eps) + eps
                t = torch.ones(shape[0], device=device) * num_t
                pred, _  = self.xstart_model(item_rep, x, t*extra, torch.ones_like(t) * extra, mask_seq)

                V = pred - z0

                x = x.detach().clone() + V * dt

            nfe = N
        return x, nfe

    def reverse_p_sample_rf(self, item_rep, z0, mask_seq, stage):
        X_pred, nfe = self.euler_sampler(item_rep, mask_seq, z0, stage)
        return X_pred

    def Sin_fn(self, t):
        half_pi = math.pi / 2
        return torch.sin(half_pi * t)

    def Cos_fn(self, t):
        half_pi = math.pi / 2
        return torch.cos(half_pi * t)


    def a_t_fn(self, t):
        return t

    def b_t_fn(self, t):
        return 1 - t

    def a_t_derivative(self, t):
        return -torch.ones_like(t)

    def b_t_derivative(self, t):
        return torch.ones_like(t)

    @property
    def T(self):
      return 1.


    def q_sample_rf(self, x_start, t, z0, mask=None):
        """
        Construct noisy sample at time t: x_t = t * x_0 + (1 - t) * noise
        """
        assert z0.shape == x_start.shape

        a_t = self.a_t_fn(t)       ### a_t = t
        b_t = self.b_t_fn(t)       ### b_t = 1 - t

        x_t = a_t * x_start + b_t* z0

        if mask == None:
            return x_t

        else:
            mask = th.broadcast_to(mask.unsqueeze(dim=-1), x_start.shape)
            return th.where(mask==0, x_start, x_t)


    # Logit-Normal Sampling function for PyTorch
    def logit_normal_sampling_torch(self, m, s, batch_size):

        u_samples = torch.normal(mean=m, std=s, size=(batch_size,))

        t_samples = 1 / (1 + torch.exp(-u_samples))
        return t_samples


    def Mode_sample_timestep(self, batch_size, s, device):

        u = torch.rand(batch_size, device=device)

        correction_term = s * (torch.cos((torch.pi / 2) * u)**2 - 1 + u)

        t = 1 - u - correction_term

        return t

    def CosMap_sample_timesteps(self, batch_size, device):

        u = torch.rand(batch_size, device=device)

        t = 1 - 1 / (torch.tan((torch.pi / 2) * u) + 1)

        return t

    def forward(self, item_rep, item_tag, mask_seq):
        noise = th.randn_like(item_tag)

        z0 = noise
        batch_size = item_tag.shape[0]

        # Mode Sampling with Heavy Tails
        if self.sampling_method == 'mode':
            t_rf = self.Mode_sample_timestep(batch_size, self.s_modsamp, item_tag.device) * (self.T - self.eps) + self.eps

        # Uniform Sampling
        elif self.sampling_method == 'uniform':
            t_rf = torch.rand(item_tag.shape[0], device=item_tag.device) * (self.T - self.eps) + self.eps

        # Logit-Normal Sampling
        elif self.sampling_method == 'logit_normal':
            t_rf = self.logit_normal_sampling_torch(self.m_logNorm, self.s_logNorm, batch_size) * (self.T - self.eps) + self.eps
            t_rf = t_rf.to(item_tag.device)

        # CosMap Sampling
        elif self.sampling_method == 'cosmap':
            t_rf = self.CosMap_sample_timesteps(batch_size, item_tag.device) * (self.T - self.eps) + self.eps

        else:
            raise ValueError(f"Unsupported sampling method: {self.sampling_method}")

        t_rf_expand = t_rf.view(-1, 1).repeat(1, item_tag.shape[1])

        x_t = self.q_sample_rf(item_tag, t_rf_expand, z0=z0)

        extra = (1 / self.eps) - 1

        #####X0_pred
        x_0, decode_out = self.xstart_model(item_rep, x_t, t_rf*extra, torch.ones_like(t_rf).detach(), mask_seq)

        return x_0, decode_out, t_rf_expand, t_rf, z0
