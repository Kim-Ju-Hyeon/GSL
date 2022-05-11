import numpy as np
import torch
import torch.nn as nn
from typing import Tuple
from utils.utils import squeeze_last_dim
from models.N_BEATS.B_Block import Block


ACTIVATIONS = ['ReLU',
               'Softplus',
               'Tanh',
               'SELU',
               'LeakyReLU',
               'PReLU',
               'Sigmoid']


class NHITSBlock(Block):
    def __init__(self, n_theta_hidden, thetas_dim, pooling_mode, n_pool_kernel_size, backcast_length=10, forecast_length=5, activation='ReLU'):
        self.l_out = int(((backcast_length - n_pool_kernel_size) / n_pool_kernel_size) + 1)
        super(NHITSBlock, self).__init__(n_theta_hidden, thetas_dim, self.l_out, forecast_length, activation)
        assert (pooling_mode in ['max', 'average'])
        self.n_pool_kernel_size = n_pool_kernel_size

        if pooling_mode == 'max':
            self.pooling_layer = nn.MaxPool1d(kernel_size=self.n_pool_kernel_size,
                                              stride=self.n_pool_kernel_size, ceil_mode=True)
        elif pooling_mode == 'average':
            self.pooling_layer = nn.AvgPool1d(kernel_size=self.n_pool_kernel_size,
                                              stride=self.n_pool_kernel_size, ceil_mode=True)


    def forward(self, x):
        x = squeeze_last_dim(x)
        x = self.pooling_layer(x)
        x = super(NHITSBlock, self).forward(x)

        theta_b = self.theta_b_fc(x)
        theta_f = self.theta_f_fc(x)




class NHITSBlock(nn.Module):
    """
    N-HiTS block which takes a basis function as an argument.
    """

    def __init__(self, n_time_in: int, n_time_out: int, n_x: int,
                 n_s: int, n_s_hidden: int, n_theta: int, n_theta_hidden: list,
                 n_pool_kernel_size: int, pooling_mode: str, basis: nn.Module,
                 n_layers: int, batch_normalization: bool, dropout_prob: float, activation: str):

        super().__init__()

        assert (pooling_mode in ['max', 'average'])

        n_time_in_pooled = int(np.ceil(n_time_in / n_pool_kernel_size))

        if n_s == 0:
            n_s_hidden = 0
        n_theta_hidden = [n_time_in_pooled + (n_time_in + n_time_out) * n_x + n_s_hidden] + n_theta_hidden

        self.n_time_in = n_time_in
        self.n_time_out = n_time_out

        self.n_s = n_s
        self.n_s_hidden = n_s_hidden

        self.n_x = n_x

        self.n_pool_kernel_size = n_pool_kernel_size
        self.batch_normalization = batch_normalization
        self.dropout_prob = dropout_prob

        assert activation in ACTIVATIONS, f'{activation} is not in {ACTIVATIONS}'
        activ = getattr(nn, activation)()

        if pooling_mode == 'max':
            self.pooling_layer = nn.MaxPool1d(kernel_size=self.n_pool_kernel_size,
                                              stride=self.n_pool_kernel_size, ceil_mode=True)
        elif pooling_mode == 'average':
            self.pooling_layer = nn.AvgPool1d(kernel_size=self.n_pool_kernel_size,
                                              stride=self.n_pool_kernel_size, ceil_mode=True)

        hidden_layers = []
        for i in range(n_layers):
            hidden_layers.append(nn.Linear(in_features=n_theta_hidden[i], out_features=n_theta_hidden[i + 1]))
            hidden_layers.append(activ)

            if self.batch_normalization:
                hidden_layers.append(nn.BatchNorm1d(num_features=n_theta_hidden[i + 1]))

            if self.dropout_prob > 0:
                hidden_layers.append(nn.Dropout(p=self.dropout_prob))

        output_layer = [nn.Linear(in_features=n_theta_hidden[-1], out_features=n_theta)]
        layers = hidden_layers + output_layer

        # n_s is computed with data, n_s_hidden is provided by user, if 0 no statics are used
        if (self.n_s > 0) and (self.n_s_hidden > 0):
            self.static_encoder = _StaticFeaturesEncoder(in_features=n_s, out_features=n_s_hidden)
        self.layers = nn.Sequential(*layers)
        self.basis = basis

    def forward(self, insample_y: torch.Tensor, insample_x_t: torch.Tensor,
                outsample_x_t: torch.Tensor, x_s: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:

        insample_y = insample_y.unsqueeze(1)
        # Pooling layer to downsample input
        insample_y = self.pooling_layer(insample_y)
        insample_y = insample_y.squeeze(1)

        batch_size = len(insample_y)
        if self.n_x > 0:
            insample_y = torch.cat((insample_y, insample_x_t.reshape(batch_size, -1)), 1)
            insample_y = torch.cat((insample_y, outsample_x_t.reshape(batch_size, -1)), 1)

        # Static exogenous
        if (self.n_s > 0) and (self.n_s_hidden > 0):
            x_s = self.static_encoder(x_s)
            insample_y = torch.cat((insample_y, x_s), 1)

        # Compute local projection weights and projection
        theta = self.layers(insample_y)
        backcast, forecast = self.basis(theta, insample_x_t, outsample_x_t)

        return backcast, forecast
