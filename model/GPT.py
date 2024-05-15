import torch
import torch.nn as nn
import math
import torch.nn.functional as F

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

class PositionWiseFeedForward(nn.Module):
    def __init__(self, embed_dim, hidden_dim, dropout=0.0):
        super().__init__()

        # Model Configurations
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout

        # Model Layers
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class TransfomerCausalEncoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, hidden_dim, dropout=0.0):
        super().__init__()

        # Model Configurations
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.dropout = dropout

        # Model Layers
        self.attn = MultiHeadAttention(embed_dim, num_heads, dropout=dropout)
        self.ff = PositionWiseFeedForward(embed_dim, hidden_dim, dropout=dropout)
        self.attn_norm = nn.LayerNorm(embed_dim)
        self.ff_norm = nn.LayerNorm(embed_dim)
        self.post_norm = nn.LayerNorm(embed_dim)

    def forward(self, x, pad_mask, causal_mask):

        x = self.attn_norm(x)
        x = x + self.attn(x, x, x, pad_mask, causal_mask)

        x = self.ff_norm(x)
        x = x + self.ff(x)

        return x

class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, embed_dim, max_length=1000):
        super().__init__()

        # Model Configurations
        self.embed_dim = embed_dim
        self.max_length = max_length

        # Model Layers
        pe = torch.zeros(max_length, embed_dim)
        position = torch.arange(0, max_length).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return x

class LearnablePositionalEncoding(nn.Module):
    def __init__(self, embed_dim, max_length=1000):
        super().__init__()

        # Model Configurations
        self.embed_dim = embed_dim
        self.max_length = max_length

        # Model Layers
        self.pe = nn.Parameter(torch.empty(max_length, embed_dim))  # Initialize an empty tensor
        nn.init.uniform_(self.pe, -0.02, 0.02)  # Fill it with values uniformly sampled from [-0.02, 0.02]

    def forward(self, x):
        x = x + self.pe[:x.size(1)]
        return x

class TSGPTEncoder(nn.Module):
    def __init__(self, input_dims, output_dims, embed_dim, num_heads, num_layers, hidden_dim, dropout=0.0, ablationtag=None, max_length=1000):
        super().__init__()

        # Model Configurations
        self.input_dims = input_dims
        self.output_dims = output_dims
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.dropout = dropout

        # Model Layers
        self.pre_proj = nn.Linear(input_dims, embed_dim)

        """if ablationtag == 'learnable_pos':
            self.positional_embedding = LearnablePositionalEncoding(embed_dim, max_length)
        elif ablationtag == 'sinusoidal_pos':
            self.positional_embedding = SinusoidalPositionalEncoding(embed_dim, max_length)
        else:
            self.positional_embedding = None"""

        self.layers = nn.ModuleList([
            TransfomerCausalEncoderLayer(embed_dim, num_heads, hidden_dim, dropout=dropout)
            for _ in range(num_layers)
        ])
        self.post_proj = nn.Linear(embed_dim, output_dims)
        self.final_proj = nn.Linear(output_dims, output_dims // 2)

        """self.preL2 = True
        self.postL2 = True
        if ablationtag == 'w/o token L2':
            self.preL2 = False
        elif ablationtag == 'w/o repr L2':
            self.postL2 = False
        elif ablationtag == 'w/o L2':
            self.preL2 = False
            self.postL2 = False"""

    def generate_causal_mask(self, x):
        mask = torch.triu(x.new_ones(x.size(1), x.size(1)), diagonal=1).to(x.device).bool()
        return mask

    def generate_pad_mask(self, x):
        return torch.isnan(x).any(dim=-1)

    def generate_random_mask(self, x, p): # Size must be (B, S)
        return torch.rand(x.size(0), x.size(1), device=x.device) > p
    def forward(self, x):

        # Generate masks
        pad_mask = self.generate_pad_mask(x)
        causal_mask = self.generate_causal_mask(x)
        combined_mask = pad_mask | self.generate_random_mask(x, 0.2) if self.training else pad_mask # Perturb 10% of the data

        x[combined_mask] = 0
        x = self.pre_proj(x)
        x = F.normalize(x, p=2, dim=-1) #if self.preL2 else x # For ablation study

        #if self.positional_embedding is not None:
        #    x = self.positional_embedding(x)

        for layer in self.layers:
            x = layer(x, combined_mask, causal_mask)

        x = self.post_proj(x)
        x = F.normalize(x, p=2, dim=-1) #if self.postL2 else x # For ablation study

        x[pad_mask] = float('nan')
        return x

    def final_projection(self, x):
        pad_mask = self.generate_pad_mask(x)
        x[pad_mask] = 0
        x = self.final_proj(x)
        x[pad_mask] = float('nan')
        return x