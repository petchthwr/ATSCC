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
        x = self.repr_dropout(self.feature_extractor(x))  # B x Co x T
        x = x.transpose(1, 2)  # B x T x Co
        #x = F.normalize(x, p=2, dim=-1)
        x[~nan_mask] = float('nan')

        return x

class TSEncoderLSTM(nn.Module):
    def __init__(self, input_dims, output_dims, hidden_dims=60, depth=3, mask_mode='binomial', dropout=0.1, all_out=False):
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
        self.lstm = nn.LSTM(hidden_dims, hidden_dims, depth, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dims, output_dims)
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
        x = self.repr_dropout(x)  # B x Co x T
        x, _ = self.lstm(x)
        x = self.fc(x)
        x[~nan_mask] = float('nan')

        return x