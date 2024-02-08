import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from .inception import InceptionBlock, Flatten, correct_sizes

class RnnEncoder(torch.nn.Module):
    def __init__(self, hidden_size, in_channel, encoding_size, cell_type='GRU', num_layers=1, device='cpu', dropout=0, bidirectional=True):
        super(RnnEncoder, self).__init__()
        self.hidden_size = hidden_size
        self.in_channel = in_channel
        self.num_layers = num_layers
        self.cell_type = cell_type
        self.encoding_size = encoding_size
        self.bidirectional = bidirectional
        self.device = device

        self.nn = torch.nn.Sequential(torch.nn.Linear(self.hidden_size*(int(self.bidirectional) + 1), self.encoding_size)).to(self.device)
        if cell_type=='GRU':
            self.rnn = torch.nn.GRU(input_size=self.in_channel, hidden_size=self.hidden_size, num_layers=num_layers,
                                    batch_first=False, dropout=dropout, bidirectional=bidirectional).to(self.device)

        elif cell_type=='LSTM':
            self.rnn = torch.nn.LSTM(input_size=self.in_channel, hidden_size=self.hidden_size, num_layers=num_layers,
                                    batch_first=False, dropout=dropout, bidirectional=bidirectional).to(self.device)
        else:
            raise ValueError('Cell type not defined, must be one of the following {GRU, LSTM, RNN}')

    def forward(self, x):
        x = x.permute(2,0,1)
        if self.cell_type=='GRU':
            past = torch.zeros(self.num_layers * (int(self.bidirectional) + 1), x.shape[1], self.hidden_size).to(self.device)
        elif self.cell_type=='LSTM':
            h_0 = torch.zeros(self.num_layers * (int(self.bidirectional) + 1), (x.shape[1]), self.hidden_size).to(self.device)
            c_0 = torch.zeros(self.num_layers * (int(self.bidirectional) + 1), (x.shape[1]), self.hidden_size).to(self.device)
            past = (h_0, c_0)
        out, _ = self.rnn(x.to(self.device), past)  # out shape = [seq_len, batch_size, num_directions*hidden_size]
        encodings = self.nn(out[-1].squeeze(0))
        return encodings


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

class Expansion(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Expansion, self).__init__()
        self.input_dim = input_dim  # Original feature dimension
        self.output_dim = output_dim  # Desired output dimension
        self.linear = nn.Linear(self.input_dim, self.output_dim)  # Linear layer
    def forward(self, x):
        N, _, S = x.shape # x should be of shape (N, input_dim, S)
        x = x.permute(0, 2, 1).reshape(-1, self.input_dim) # Reshape and permute the tensor to fit the input requirements of nn.Linear
        x = self.linear(x) # Apply the linear transformation
        x = x.reshape(N, S, self.output_dim).permute(0, 2, 1) # Reshape and permute back to get the tensor in desired shape (N, output_dim, S)
        return x

class TransformerEncoder(nn.Module):
    def __init__(self, input_size, d_model, nhead, num_layers, dim_feedforward=512, dropout=0.1, encoding_size=15, device='cuda'):
        super(TransformerEncoder, self).__init__()

        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.encoding_size = encoding_size
        self.device = device
        self.max_position_encoding = 100

        self.cls_token = nn.Parameter(torch.randn(1, d_model, 1)).to(device) # Introducing a learnable token to the input
        self.projection = Expansion(input_size, d_model).to(device) # Expanding the input to the desired dimension

        #self.pos_encoder = PositionalEncoding(d_model, dropout).to(device) # Positional encoding

        # Implement learnable positional encoding
        self.pos_encoder = nn.Embedding(self.max_position_encoding, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, activation='gelu').to(device) # The transformer encoder layer
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers).to(device) # The transformer encoder
        self.fcn = nn.Linear(d_model, encoding_size).to(device)  # Modify the output size of the linear layer
        self.feature_norm = nn.LayerNorm(encoding_size).to(device)

    def forward(self, x):
        x = self.projection(x) # Expanding the input to the desired dimension
        cls_tokens = self.cls_token.repeat(x.shape[0], 1, 1)  # Adding the token to the beginning of every sequence in the batch, Shape: (N, E, 1)
        x = torch.cat([cls_tokens, x], dim=2).to(self.device)  # Shape: (N, E, S+1)
        x = x.permute(2, 0, 1)  # Permuted to (S+1, N, E)
        x = x + self.pos_encoder(torch.arange(x.size(0)).to(self.device)).unsqueeze(1)  # Apply positional encoding
        encodings = self.encoder(x) # Apply the transformer encoder
        token_representations = encodings[0] #Extracting the token's representation after the encoder, Shape: (N, E)
        encodings = self.fcn(token_representations) # Apply the FCN to map to the desired output size
        encodings = self.feature_norm(encodings)

        return encodings

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
        self.max_position_encoding = 2048

        #self.cls_token = nn.Parameter(torch.randn(1, 1, d_model)).to(device) # Introducing a learnable token to the input
        self.projection = nn.Linear(input_dim, d_model).to(device) # Expanding the input to the desired dimension
        #self.pos_encoder = nn.Parameter(torch.randn(1, self.max_position_encoding, d_model)).to(device) # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, dropout).to(device) # Positional encoding
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, activation='gelu', batch_first=True).to(device) # The transformer encoder layer
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers).to(device) # The transformer encoder
        self.fcn = nn.Linear(d_model, output_dim).to(device)  # Modify the output size of the linear layer

    def generate_padding_mask(self, x):
        # x is of shape (B, T, Ch)
        padding_mask = torch.isnan(x).any(dim=-1).to(x.device) # Shape: (B, T)
        return padding_mask

    def generate_causal_mask(self, x):
        # x is of shape (B, T, Ch)
        T = x.size(1)
        causal_mask = torch.triu(torch.ones((T, T), dtype=torch.bool), diagonal=1).to(x.device) # Shape: (T, T)
        return causal_mask

    def generate_binomial_mask(self, x):
        # x is of shape (B, T, Ch)
        mask = generate_binomial_mask(x.size(0), x.size(1)).to(x.device)
        return mask

    def forward(self, x):

        mask = self.generate_padding_mask(x)
        x[mask] = 0
        x = self.projection(x) # (N, S, E)
        #cls_tokens = self.cls_token.repeat(x.shape[0], 1, 1) # (N, 1, E)
        #x = torch.cat([cls_tokens, x], dim=1).to(self.device) # (N, S+1, E)

        #x = x + self.pos_encoder[:, :x.size(1), :] # (N, S+1, E) + (1, S+1, E) = (N, S+1, E)
        x = self.pos_encoder(x.permute(1, 0, 2)).permute(1, 0, 2) # (N, S+1, E)
        #mask = torch.cat([torch.zeros(mask.shape[0], 1, dtype=torch.bool).to(mask.device), mask], dim=1)

        # Apply the transformer encoder
        x = self.encoder(x, src_key_padding_mask=mask)
        x = self.fcn(x)

        # Apply NaN back to the output
        x[mask] = float('nan')

        return x

class InceptionTimeEncoder(nn.Module):
    def __init__(self, input_dim, d_model, n_filters=32, kernel_sizes=[5, 11, 23], bottleneck_channels=32,
                 use_residual=True, activation_function=nn.GELU, num_blocks=3, encoding_size=128,
                 dropout_rate=0.5, use_batchnorm=True, device='cpu'):

        super(InceptionTimeEncoder, self).__init__()

        # Store configurations
        kernel_sizes = correct_sizes(kernel_sizes)
        self.input_dim = input_dim
        self.d_model = d_model
        self.n_filters = n_filters
        self.kernel_sizes = kernel_sizes
        self.bottleneck_channels = bottleneck_channels
        self.use_residual = use_residual
        self.activation_function = activation_function
        self.num_blocks = num_blocks
        self.encoding_size = encoding_size
        self.dropout_rate = dropout_rate
        self.use_batchnorm = use_batchnorm
        self.device = device

        # Expansion layer
        self.expansion = Expansion(self.input_dim, self.d_model)

        # Activation function
        self.activation = activation_function()

        # Set up the inception blocks
        modules = []
        in_channels = self.d_model

        for _ in range(num_blocks):
            modules.append(
                InceptionBlock(
                    in_channels=in_channels,
                    n_filters=n_filters,
                    kernel_sizes=kernel_sizes,
                    bottleneck_channels=bottleneck_channels,
                    use_residual=use_residual,
                    activation=self.activation
                )
            )

            if use_batchnorm:
                modules.append(nn.BatchNorm1d(num_features=n_filters * 4))

            modules.append(nn.Dropout(p=dropout_rate))

            in_channels = n_filters * 4

        self.inception_blocks = nn.Sequential(*modules)
        self.adaptive_avg_pool = nn.AdaptiveAvgPool1d(output_size=1)
        self.flatten = Flatten(out_features=in_channels)
        self.linear = nn.Linear(in_features=in_channels, out_features=encoding_size)

        # Send model to specified device
        self.to(device)

    def forward(self, x):
        x = self.expansion(x)
        x = self.inception_blocks(x)
        x = self.adaptive_avg_pool(x)
        x = self.flatten(x)
        x = self.linear(x)
        return x


class InceptionExpand(nn.Module):
    def __init__(self, input_dim, n_filters=32, kernel_sizes=[5, 11, 23], bottleneck_channels=32,
                 use_residual=True, activation_function=nn.GELU, num_blocks=3,
                 dropout_rate=0.5, use_batchnorm=True, device='cpu'):

        super(InceptionExpand, self).__init__()

        kernel_sizes = correct_sizes(kernel_sizes)

        # Activation function
        self.activation = activation_function()

        # Set up the inception blocks for expansion
        modules = []
        in_channels = input_dim

        for _ in range(num_blocks):
            modules.append(
                InceptionBlock(
                    in_channels=in_channels,
                    n_filters=n_filters,
                    kernel_sizes=kernel_sizes,
                    bottleneck_channels=bottleneck_channels,
                    use_residual=use_residual,
                    activation=self.activation
                )
            )

            if use_batchnorm:
                modules.append(nn.BatchNorm1d(num_features=n_filters * 4))

            modules.append(nn.Dropout(p=dropout_rate))

            in_channels = n_filters * 4

        self.inception_blocks = nn.Sequential(*modules)
        self.to(device)

    def forward(self, x):
        x = self.inception_blocks(x) # get (N, E, S), out (N, 4*n_filters, S)
        return x


class TransformerWithInceptionEncoder(nn.Module):
    def __init__(self, input_size, d_model, nhead, num_layers, dim_feedforward=512, dropout=0.1, encoding_size=15,
                 inception_kernel_sizes=[5, 11, 23], inception_bottleneck_channels=16,
                 use_residual=True, activation_function=nn.GELU, num_blocks=3,
                 inception_dropout_rate=0.5, use_batchnorm=True, device='cpu'):
        super(TransformerWithInceptionEncoder, self).__init__()

        # Transformer Configurations
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.encoding_size = encoding_size
        self.device = device

        # Inception Configurations
        self.inception_kernel_sizes = correct_sizes(inception_kernel_sizes)
        self.inception_bottleneck_channels = inception_bottleneck_channels
        self.use_residual = use_residual
        self.activation_function = activation_function
        self.num_blocks = num_blocks
        self.inception_dropout_rate = inception_dropout_rate
        self.use_batchnorm = use_batchnorm

        # InceptionExpand layer
        self.inception_expand = InceptionExpand(
            input_dim=input_size,
            n_filters= self.d_model // 4,
            kernel_sizes=self.inception_kernel_sizes,
            bottleneck_channels=self.inception_bottleneck_channels,
            use_residual=self.use_residual,
            activation_function=self.activation_function,
            num_blocks=self.num_blocks,
            dropout_rate=self.inception_dropout_rate,
            use_batchnorm=self.use_batchnorm
        ).to(device)

        self.cls_token = nn.Parameter(torch.randn(1, d_model, 1)).to(device)  # Introducing a learnable token to the input
        self.pos_encoder = PositionalEncoding(d_model, dropout).to(device)  # Positional encoding
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, activation='gelu').to(device)  # The transformer encoder layer
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers).to(device)  # The transformer encoder
        self.fcn = nn.Linear(d_model, encoding_size).to(device)  # Modify the output size of the linear layer

    def forward(self, x):
        x = self.inception_expand(x)  # Using InceptionExpand for input projection
        cls_tokens = self.cls_token.repeat(x.shape[0], 1, 1)  # Adding the token to the beginning of every sequence in the batch, Shape: (N, E, 1)
        x = torch.cat([cls_tokens, x], dim=2).to(self.device)  # Shape: (N, E, S+1)
        x = x.permute(2, 0, 1)  # Permuted to (S+1, N, E)
        x = self.pos_encoder(x)  # Apply positional encoding
        encodings = self.encoder(x)  # Apply the transformer encoder
        token_representations = encodings[0]  # Extracting the token's representation after the encoder, Shape: (N, E)
        encodings = self.fcn(token_representations)  # Apply the FCN to map to the desired output size

        return encodings


class MimicEncoder(torch.nn.Module):
    def __init__(self, input_size, in_channel, encoding_size):
        super(MimicEncoder, self).__init__()
        self.input_size = input_size
        self.in_channel = in_channel
        self.encoding_size = encoding_size

        self.nn = torch.nn.Sequential(torch.nn.Linear(input_size, 64),
                                      torch.nn.Dropout(),
                                      torch.nn.ReLU(),
                                      torch.nn.Linear(64, encoding_size))

    def forward(self, x):
        x = torch.mean(x, dim=1)
        encodings = self.nn(x)
        return encodings


class WFEncoder(nn.Module):
    def __init__(self, encoding_size, classify=False, n_classes=None):
        # Input x is (batch, 2, 256)
        super(WFEncoder, self).__init__()

        self.encoding_size = encoding_size
        self.n_classes = n_classes
        self.classify = classify
        self.classifier =None
        if self.classify:
            if self.n_classes is None:
                raise ValueError('Need to specify the number of output classes for te encoder')
            else:
                self.classifier = nn.Sequential(
                    nn.Dropout(0.5),
                    nn.Linear(self.encoding_size, self.n_classes)
                )
                nn.init.xavier_uniform_(self.classifier[1].weight)

        self.features = nn.Sequential(
            nn.Conv1d(2, 64, kernel_size=4, stride=1, padding=1),
            nn.ELU(inplace=True),
            nn.BatchNorm1d(64, eps=0.001),
            nn.Conv1d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ELU(inplace=True),
            nn.BatchNorm1d(64, eps=0.001),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ELU(inplace=True),
            nn.BatchNorm1d(128, eps=0.001),
            # nn.Dropout(),
            nn.Conv1d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ELU(inplace=True),
            nn.BatchNorm1d(128, eps=0.001),
            nn.MaxPool1d(kernel_size=2, stride=2),
            # nn.Dropout(0.5),
            nn.Conv1d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ELU(inplace=True),
            nn.BatchNorm1d(256, eps=0.001),
            # nn.Dropout(0.5),
            nn.Conv1d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ELU(inplace=True),
            nn.BatchNorm1d(256, eps=0.001),
            nn.MaxPool1d(kernel_size=2, stride=2)
            )

        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(79872, 2048),
            nn.ELU(inplace=True),
            nn.BatchNorm1d(2048, eps=0.001),
            nn.Linear(2048, self.encoding_size)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        encoding = self.fc(x)
        if self.classify:
            c = self.classifier(encoding)
            return c
        else:
            return encoding


class StateClassifier(torch.nn.Module):
    def __init__(self, input_size, output_size):
        super(StateClassifier, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.normalize = torch.nn.BatchNorm1d(self.input_size)
        self.nn = torch.nn.Linear(self.input_size, self.output_size)
        torch.nn.init.xavier_uniform_(self.nn.weight)

    def forward(self, x):
        x = self.normalize(x)
        logits = self.nn(x)
        return logits


class WFClassifier(torch.nn.Module):
    def __init__(self, encoding_size, output_size):
        super(WFClassifier, self).__init__()
        self.encoding_size = encoding_size
        self.output_size = output_size
        self.classifier = nn.Linear(self.encoding_size, output_size)
        torch.nn.init.xavier_uniform_(self.classifier.weight)

    def forward(self, x):
        c = self.classifier(x)
        return c


class E2EStateClassifier(torch.nn.Module):
    def __init__(self, hidden_size, in_channel, encoding_size, output_size, cell_type='GRU', num_layers=1, dropout=0,
                 bidirectional=True, device='cpu'):
        super(E2EStateClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.in_channel = in_channel
        self.num_layers = num_layers
        self.cell_type = cell_type
        self.encoding_size = encoding_size
        self.bidirectional = bidirectional
        self.output_size = output_size
        self.device = device

        self.fc = torch.nn.Sequential(torch.nn.Linear(self.hidden_size*(int(self.bidirectional) + 1), self.encoding_size)).to(self.device)
        self.nn = torch.nn.Sequential(torch.nn.Linear(self.encoding_size, self.output_size)).to(self.device)
        if cell_type=='GRU':
            self.rnn = torch.nn.GRU(input_size=self.in_channel, hidden_size=self.hidden_size, num_layers=num_layers,
                                    batch_first=False, dropout=dropout, bidirectional=bidirectional).to(self.device)
        elif cell_type=='LSTM':
            self.rnn = torch.nn.LSTM(input_size=self.in_channel, hidden_size=self.hidden_size, num_layers=num_layers,
                                    batch_first=False, dropout=dropout, bidirectional=bidirectional).to(self.device)
        else:
            raise ValueError('Cell type not defined, must be one of the following {GRU, LSTM, RNN}')

    def forward(self, x):
        x = x.permute(2,0,1)
        if self.cell_type=='GRU':
            past = torch.zeros(self.num_layers * (int(self.bidirectional) + 1), x.shape[1], self.hidden_size).to(self.device)
        elif self.cell_type=='LSTM':
            h_0 = torch.zeros(self.num_layers * (int(self.bidirectional) + 1), (x.shape[1]), self.hidden_size).to(self.device)
            c_0 = torch.zeros(self.num_layers * (int(self.bidirectional) + 1), (x.shape[1]), self.hidden_size).to(self.device)
            past = (h_0, c_0)
        out, _ = self.rnn(x, past)  # out shape = [seq_len, batch_size, num_directions*hidden_size]
        encodings = self.fc(out[-1].squeeze(0))
        return self.nn(encodings)