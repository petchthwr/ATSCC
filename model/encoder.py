import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import math
from .dilated_conv import DilatedConvEncoder

def generate_continuous_mask(B, T, n=5, l=0.1):
    res = torch.full((B, T), True, dtype=torch.bool)
    if isinstance(n, float):
        n = int(n * T)
    n = max(min(n, T // 2), 1)
    
    if isinstance(l, float):
        l = int(l * T)
    l = max(l, 1)
    
    for i in range(B):
        for _ in range(n):
            t = np.random.randint(T-l+1)
            res[i, t:t+l] = False
    return res

def generate_binomial_mask(B, T, p=0.5):
    return torch.from_numpy(np.random.binomial(1, p, size=(B, T))).to(torch.bool)

class TSEncoder(nn.Module):
    def __init__(self, input_dims, output_dims, hidden_dims=64, depth=10, mask_mode='binomial', dropout=0.1, all_out=False):
        super().__init__()
        # Model Configurations
        self.input_dims = input_dims
        self.output_dims = output_dims
        self.hidden_dims = hidden_dims
        self.depth = depth
        self.mask_mode = mask_mode
        self.dropout = dropout
        self.all_out = all_out
        self.conv_pretrained = False

        # Feature Extractor
        self.input_fc = nn.Linear(input_dims, hidden_dims)
        self.feature_extractor = DilatedConvEncoder(
            hidden_dims,
            [hidden_dims] * depth + [output_dims],
            kernel_size=3,
            dropout=dropout
        )
        self.repr_dropout = nn.Dropout(p=dropout)
        self.pre_layer_norm = nn.LayerNorm(hidden_dims)

    def forward(self, x, mask=None):  # x: B x T x input_dims
        nan_mask = ~x.isnan().any(axis=-1)
        x[~nan_mask] = 0
        x = self.input_fc(x)  # B x T x Ch

        # generate & apply mask
        if mask is None:
            if self.training:
                mask = self.mask_mode
            else:
                mask = 'all_true'

        if mask == 'binomial':
            mask = generate_binomial_mask(x.size(0), x.size(1)).to(x.device)
        elif mask == 'continuous':
            mask = generate_continuous_mask(x.size(0), x.size(1)).to(x.device)
        elif mask == 'all_true':
            mask = x.new_full((x.size(0), x.size(1)), True, dtype=torch.bool)
        elif mask == 'all_false':
            mask = x.new_full((x.size(0), x.size(1)), False, dtype=torch.bool)
        elif mask == 'mask_last':
            mask = x.new_full((x.size(0), x.size(1)), True, dtype=torch.bool)
            mask[:, -1] = False

        mask &= nan_mask
        x[~mask] = 0

        # conv encoder
        x = x.transpose(1, 2)  # B x Ch x T
        if not self.all_out:
            # Process with the entire feature extractor network
            x = self.repr_dropout(self.feature_extractor(x))  # B x Co x T
            x = x.transpose(1, 2)  # B x T x Co
            x = F.normalize(x, p=2, dim=-1)
            # Masking NAN back
            x[~nan_mask] = float('nan')

            return x
        else:
            outputs = []
            x = self.repr_dropout(x)
            for layer in self.feature_extractor.net:
                x = layer(x)
                outputs.append(x.transpose(1, 2))

            # Masking NAN back
            for i in range(len(outputs)):
                outputs[i][~nan_mask] = float('nan')

            return outputs

class WrapUpTSEncoder(nn.Module):
    def __init__(self, input_dims, output_dims, hidden_dims=64, depth=10, mask_mode='binomial'):
        super().__init__()
        # Initialize the TSEncoder inside the wrapper
        self.encoding_size = output_dims
        self.ts_encoder = TSEncoder(input_dims, output_dims, hidden_dims, depth, mask_mode)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.ts_encoder(x)
        z, _ = torch.max(x, dim=1)
        return z


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.0, bias=False):
        super().__init__()

        # Model Configurations
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout

        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        # Model Layers
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.dropout = nn.Dropout(dropout)
        self.softmax = nn.Softmax(dim=-1)

    def split_heads(self, x):
        batch_size, seq_len, embed_dim = x.size()  # x: (B, S, E)
        return x.view(batch_size, seq_len, self.num_heads, embed_dim // self.num_heads).permute(0, 2, 1, 3)  # x: (B, S, H, D) to (B, H, S, D)

    def scaled_dot_product(self, q, k, v, pad_mask, causal_mask):

        # QKV Operations
        score = torch.einsum('bhqd, bhkd -> bhqk', q, k) / math.sqrt(self.embed_dim)  # (B, H, S, D) * (B, H, S, D) -> (B, H, S, S)

        # Mask the score matrix
        if causal_mask is not None:
            score = score.masked_fill(causal_mask.unsqueeze(0).unsqueeze(1), -1e20)  # (1, 1, S, S)

        if pad_mask is not None:
            score = score.masked_fill(pad_mask.unsqueeze(1).unsqueeze(2), -1e20)  # (B, 1, 1, S)

        attn_weights = self.softmax(score)  # (B, H, S, S)
        attn_weights = self.dropout(attn_weights)  # (B, H, S, S) # First dropout on attention weights
        attn_output = torch.einsum('bhqk, bhkd -> bhqd', attn_weights, v)  # (B, H, S, S) * (B, H, S, D) -> (B, H, S, D)

        # Concatenate the heads and project to the output dimension
        attn_output = attn_output.contiguous().permute(0, 2, 1, 3).reshape(attn_output.size(0), -1, self.embed_dim)  # (B, H, S, D) -> (B, S, H, D) -> (B, S, E)
        attn_output = self.out_proj(attn_output)  # (B, S, E) -> (B, S, E)
        attn_output = self.dropout(attn_output)  # (B, S, E) # Second dropout on attention output

        return attn_output, attn_weights

    def forward(self, q, k, v, pad_mask, causal_mask):

        q, k, v = self.q_proj(q), self.k_proj(k), self.v_proj(v)  # (B, S, E) -> (B, S, E)
        q, k, v = self.split_heads(q), self.split_heads(k), self.split_heads(v)  # (B, S, E) -> (B, H, S, D)
        attn_output, attn_weights = self.scaled_dot_product(q, k, v, pad_mask, causal_mask)  # (B, S, E), (B, H, S, S)

        return attn_output