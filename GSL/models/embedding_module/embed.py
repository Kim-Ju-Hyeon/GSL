import torch
import torch.nn as nn
import torch.nn.functional as F
import re
import math


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(-1)]


class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        padding = 1
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model,
                                   kernel_size=3, padding=padding, padding_mode='zeros')
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        x = self.tokenConv(x).transpose(1, 2)
        return x


class TimeFeatureEmbedding(nn.Module):
    def __init__(self, d_model, freq='h'):
        super(TimeFeatureEmbedding, self).__init__()
        freq = re.split('(\d+)', freq)[-1].lower()

        freq_map = {'h': 4, 't': 5, 's': 6, 'min': 4, 'a': 1, 'w': 2, 'd': 3, 'b': 3}
        d_inp = freq_map[freq]
        self.embed = nn.Linear(d_inp, d_model)

    def forward(self, x):
        return self.embed(x)


class DataEmbedding(nn.Module):
    def __init__(self, c_in, embedding_dim, batch_size, nodes_num, freq='h', dropout=0.1):
        super(DataEmbedding, self).__init__()
        self.embedding_dim = embedding_dim
        self.batch_size = batch_size
        self.nodes_num = nodes_num
        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=embedding_dim)
        self.position_embedding = PositionalEmbedding(d_model=embedding_dim)
        self.temporal_embedding = TimeFeatureEmbedding(d_model=embedding_dim, freq=freq)

        self.embed_1 = nn.Linear(self.embedding_dim, self.embedding_dim // 4)
        self.embed_2 = nn.Linear(self.embedding_dim // 4, 1)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, time_stamp):
        if len(x.shape) == 3:
            _, _, L = x.size()
        else:
            _, L = x.size()
            x = x.unsqueeze(dim=1)

        x = (self.value_embedding(x).view(self.batch_size, self.nodes_num, L, self.embedding_dim) +
             self.position_embedding(x).unsqueeze(dim=0) +
             self.temporal_embedding(time_stamp.view(self.batch_size, 4, L).permute(0, 2, 1)).expand(self.nodes_num,
                                                                                                     self.batch_size,
                                                                                                     L,
                                                                                                     self.embedding_dim).permute(
                 1, 0, 2, 3)).view(-1, L, self.embedding_dim)

        x = self.dropout(self.embed_1(x))
        x = self.dropout(self.embed_2(x))

        return x
