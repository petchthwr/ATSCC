import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math



class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


def generate_binomial_mask(B, T, p=0.5):
    return torch.from_numpy(np.random.binomial(1, p, size=(B, T))).to(torch.bool)

class TransformerTSEncoder(nn.Module):
    def __init__(self, input_dim, output_dim, d_model, nhead, num_layers, dim_feedforward=512, dropout=0.1, device='cpu'):
        super(TransformerTSEncoder, self).__init__()

        # Input size (B, T, Ch)

        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.encoding_size = output_dim
        self.device = device

        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model)).to(device) # Introducing a learnable token to the input
        self.projection = nn.Linear(input_dim, d_model).to(device) # Expanding the input to the desired dimension
        self.pos_encoder = PositionalEncoding(d_model, dropout).to(device) # Positional encoding
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, activation='gelu', batch_first=True).to(device) # The transformer encoder layer
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers).to(device) # The transformer encoder
        self.fcn = nn.Linear(d_model, output_dim).to(device)  # Modify the output size of the linear layer

    def generate_padding_mask(self, x):
        # x is of shape (B, T, Ch)
        mask = torch.isnan(x).any(dim=-1).to(x.device)
        return mask

    def generate_binomial_mask(self, x):
        # x is of shape (B, T, Ch)
        mask = generate_binomial_mask(x.size(0), x.size(1)).to(x.device)
        return mask

    def forward(self, x):

        mask = self.generate_padding_mask(x)
        x[mask] = 0

        x = self.projection(x) # (N, S, E)
        cls_tokens = self.cls_token.repeat(x.shape[0], 1, 1) # (N, 1, E)
        x = torch.cat([cls_tokens, x], dim=1).to(self.device) # (N, S+1, E)
        x = self.pos_encoder(x.permute(1, 0, 2)).permute(1, 0, 2) # (N, S+1, E)
        mask = torch.cat([torch.zeros(mask.shape[0], 1, dtype=torch.bool).to(mask.device), mask], dim=1)

        # Apply the transformer encoder
        x = self.encoder(x, src_key_padding_mask=mask)
        x = self.fcn(x)

        # Apply NaN back to the output
        x[mask] = float('nan')
        cls_token = x[:, 0, :]

        return x, cls_token


# Define the parameters for the test case
B, T, Ch = 2, 10, 5  # Batch size, Sequence length, Channels
input_dim, output_dim = Ch, 7
d_model, nhead, num_layers = 512, 8, 3

# Create a random input tensor with NaN values
x = torch.randn(B, T, Ch)
x[0, 0, :] = float('nan')  # Introduce NaN value for testing

# Initialize the TransformerTSEncoder
model = TransformerTSEncoder(input_dim, output_dim, d_model, nhead, num_layers)
model.eval()

# Forward pass
output, cls_token = model(x)

print('Input shape: ', x.shape)
print('Output shape: ', output.shape)
print('CLS token shape: ', cls_token.shape)