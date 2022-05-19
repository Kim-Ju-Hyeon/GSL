import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from utils.utils import squeeze_last_dim

from torch_geometric.nn import GCNConv

ACTIVATIONS = ['ReLU',
               'Softplus',
               'Tanh',
               'SELU',
               'LeakyReLU',
               'PReLU',
               'Sigmoid']


class _TrendGenerator(nn.Module):
    def __init__(self, expansion_coefficient_dim, target_length):
        super().__init__()

        # basis is of size (expansion_coefficient_dim, target_length)
        basis = torch.stack(
            [
                (torch.arange(target_length) / target_length) ** i
                for i in range(expansion_coefficient_dim)
            ],
            dim=1,
        ).T

        self.basis = nn.Parameter(basis, requires_grad=False)

    def forward(self, x):
        return torch.matmul(x, self.basis)


class _SeasonalityGenerator(nn.Module):
    def __init__(self, target_length):
        super().__init__()
        half_minus_one = int(target_length / 2 - 1)
        cos_vectors = [
            torch.cos(torch.arange(target_length) / target_length * 2 * np.pi * i)
            for i in range(1, half_minus_one + 1)
        ]
        sin_vectors = [
            torch.sin(torch.arange(target_length) / target_length * 2 * np.pi * i)
            for i in range(1, half_minus_one + 1)
        ]

        # basis is of size (2 * int(target_length / 2 - 1) + 1, target_length)
        basis = torch.stack(
            [torch.ones(target_length)] + cos_vectors + sin_vectors, dim=1
        ).T

        self.basis = nn.Parameter(basis, requires_grad=False)

    def forward(self, x):
        return torch.matmul(x, self.basis)


class Inter_Correlation_Block(nn.Module):
    def __init__(self, n_theta_hidden, thetas_dim, backcast_length=10, forecast_length=5, activation='ReLU',
                 pooling_length=None):
        super(Inter_Correlation_Block, self).__init__()
        self.n_theta_hidden = n_theta_hidden
        self.thetas_dim = thetas_dim
        self.n_layers = len(n_theta_hidden)

        assert activation in ACTIVATIONS, f'{activation} is not in {ACTIVATIONS}'
        self.activ = getattr(nn, activation)()

        self.backcast_length = backcast_length
        self.forecast_length = forecast_length

        self.Correlation_stack = nn.ModuleList()
        for i in range(self.n_layers):
            if (i == 0) and (pooling_length is None):
                self.Correlation_stack.append(GCNConv(backcast_length, self.n_theta_hidden[i]))
            elif (i == 0) and (pooling_length is not None):
                self.Correlation_stack.append(GCNConv(pooling_length, self.n_theta_hidden[i]))
            else:
                self.Correlation_stack.append(GCNConv(self.n_theta_hidden[i - 1], self.n_theta_hidden[i]))

        self.theta_b_fc = nn.Linear(n_theta_hidden[-1], thetas_dim[0], bias=False)
        self.theta_f_fc = nn.Linear(n_theta_hidden[-1], thetas_dim[1], bias=False)

    def forward(self, x, edge_index, edge_weight=None):
        x = squeeze_last_dim(x)

        for layer in self.Correlation_stack:
            x = layer(x, edge_index, edge_weight)
            x = self.activ(x)

        return x


class GNN_SeasonalityBlock(Inter_Correlation_Block):
    def __init__(self, n_theta_hidden, thetas_dim, backcast_length=10, forecast_length=5, activation='ReLU'):
        super(GNN_SeasonalityBlock, self).__init__(n_theta_hidden, thetas_dim, backcast_length, forecast_length,
                                                   activation)
        self.backcast_seasonality_model = _SeasonalityGenerator(backcast_length)
        self.forecast_seasonality_model = _SeasonalityGenerator(forecast_length)

    def forward(self, x, edge_index, edge_weight=None):
        x = super(GNN_SeasonalityBlock, self).forward(x, edge_index, edge_weight)
        backcast = self.backcast_seasonality_model(self.theta_b_fc(x))
        forecast = self.forecast_seasonality_model(self.theta_f_fc(x))
        return backcast, forecast


class GNN_TrendBlock(Inter_Correlation_Block):
    def __init__(self, n_theta_hidden, thetas_dim, backcast_length=10, forecast_length=5, activation='ReLU'):
        super(GNN_TrendBlock, self).__init__(n_theta_hidden, thetas_dim, backcast_length, forecast_length, activation)
        self.backcast_trend_model = _TrendGenerator(thetas_dim[0], backcast_length)
        self.forecast_trend_model = _TrendGenerator(thetas_dim[1], forecast_length)

    def forward(self, x, edge_index, edge_weight=None):
        x = super(GNN_TrendBlock, self).forward(x, edge_index, edge_weight)
        backcast = self.backcast_trend_model(self.theta_b_fc(x))
        forecast = self.forecast_trend_model(self.theta_f_fc(x))
        return backcast, forecast


class GNN_GenericBlock(Inter_Correlation_Block):
    def __init__(self, n_theta_hidden, thetas_dim, backcast_length=10, forecast_length=5, activation='ReLU'):
        super(GNN_GenericBlock, self).__init__(n_theta_hidden, thetas_dim, backcast_length, forecast_length, activation)

        self.backcast_fc = nn.Linear(thetas_dim[0], backcast_length)
        self.forecast_fc = nn.Linear(thetas_dim[1], forecast_length)

    def forward(self, x, edge_index, edge_weight=None):
        x = super(GNN_GenericBlock, self).forward(x, edge_index, edge_weight)

        theta_b = self.theta_b_fc(x)
        theta_f = self.theta_f_fc(x)

        backcast = self.backcast_fc(theta_b)
        forecast = self.forecast_fc(theta_f)

        return backcast, forecast


class GNN_NHITSBlock(Inter_Correlation_Block):
    def __init__(self, n_theta_hidden, thetas_dim, pooling_mode, n_pool_kernel_size, backcast_length=10,
                 forecast_length=5, activation='ReLU'):
        self.backcast_length = backcast_length
        self.forecast_length = forecast_length
        self.l_out = int(((backcast_length - n_pool_kernel_size) / n_pool_kernel_size) + 1)

        super(GNN_NHITSBlock, self).__init__(n_theta_hidden=n_theta_hidden, thetas_dim=thetas_dim,
                                             backcast_length=self.backcast_length, forecast_length=self.forecast_length,
                                             activation=activation, pooling_length=self.l_out)
        assert (pooling_mode in ['max', 'average'])

        self.n_pool_kernel_size = n_pool_kernel_size

        if pooling_mode == 'max':
            self.pooling_layer = nn.MaxPool1d(kernel_size=self.n_pool_kernel_size,
                                              stride=self.n_pool_kernel_size, ceil_mode=True)
        elif pooling_mode == 'average':
            self.pooling_layer = nn.AvgPool1d(kernel_size=self.n_pool_kernel_size,
                                              stride=self.n_pool_kernel_size, ceil_mode=True)

    def forward(self, x, edge_index, edge_weight=None):
        x = squeeze_last_dim(x)
        x = self.pooling_layer(x)
        x = super(GNN_NHITSBlock, self).forward(x, edge_index, edge_weight)

        theta_b = self.theta_b_fc(x)
        theta_f = self.theta_f_fc(x)

        backcast = F.interpolate(theta_b[:, None, :], size=self.backcast_length,
                                 mode='linear', align_corners=False).squeeze(dim=1)
        forecast = F.interpolate(theta_f[:, None, :], size=self.forecast_length,
                                 mode='linear', align_corners=False).squeeze(dim=1)

        return backcast, forecast


class Block(nn.Module):
    def __init__(self, n_theta_hidden, thetas_dim, backcast_length=10, forecast_length=5, activation='ReLU'):
        super(Block, self).__init__()
        self.n_theta_hidden = n_theta_hidden
        self.thetas_dim = thetas_dim
        self.n_layers = len(n_theta_hidden)

        assert activation in ACTIVATIONS, f'{activation} is not in {ACTIVATIONS}'
        self.activ = getattr(nn, activation)()

        self.backcast_length = backcast_length
        self.forecast_length = forecast_length

        self.MLP_stack = nn.ModuleList()
        for i in range(self.n_layers):
            if i == 0:
                self.MLP_stack.append(nn.Linear(backcast_length, self.n_theta_hidden[i]))
            else:
                self.MLP_stack.append(nn.Linear(self.n_theta_hidden[i - 1], self.n_theta_hidden[i]))

        self.theta_b_fc = nn.Linear(n_theta_hidden[-1], thetas_dim, bias=False)
        self.theta_f_fc = nn.Linear(n_theta_hidden[-1], thetas_dim, bias=False)

    def forward(self, x):
        x = squeeze_last_dim(x)
        for layer in self.MLP_stack:
            x = layer(x)
            x = self.activ(x)
        return x


class SeasonalityBlock(Block):
    def __init__(self, n_theta_hidden, thetas_dim, backcast_length=10, forecast_length=5, activation='ReLU'):
        super(SeasonalityBlock, self).__init__(n_theta_hidden, thetas_dim, backcast_length, forecast_length, activation)
        self.backcast_seasonality_model = _SeasonalityGenerator(backcast_length)
        self.forecast_seasonality_model = _SeasonalityGenerator(forecast_length)

    def forward(self, x):
        x = super(SeasonalityBlock, self).forward(x)
        backcast = self.backcast_seasonality_model(self.theta_b_fc(x), self.backcast_linspace)
        forecast = self.forecast_seasonality_model(self.theta_f_fc(x), self.forecast_linspace)
        return backcast, forecast


class TrendBlock(Block):
    def __init__(self, n_theta_hidden, thetas_dim, backcast_length=10, forecast_length=5, activation='ReLU'):
        super(TrendBlock, self).__init__(n_theta_hidden, thetas_dim, backcast_length, forecast_length, activation)
        self.backcast_trend_model = _TrendGenerator(thetas_dim[0], backcast_length)
        self.forecast_trend_model = _TrendGenerator(thetas_dim[1], forecast_length)

    def forward(self, x):
        x = super(TrendBlock, self).forward(x)
        backcast = self.backcast_trend_model(self.theta_b_fc(x))
        forecast = self.forecast_trend_model(self.theta_f_fc(x))
        return backcast, forecast


class GenericBlock(Block):
    def __init__(self, n_theta_hidden, thetas_dim, backcast_length=10, forecast_length=5, activation='ReLU'):
        super(GenericBlock, self).__init__(n_theta_hidden, thetas_dim, backcast_length, forecast_length, activation)

        self.backcast_fc = nn.Linear(thetas_dim, backcast_length)
        self.forecast_fc = nn.Linear(thetas_dim, forecast_length)

    def forward(self, x):
        x = super(GenericBlock, self).forward(x)

        theta_b = self.theta_b_fc(x)
        theta_f = self.theta_f_fc(x)

        backcast = self.backcast_fc(theta_b)
        forecast = self.forecast_fc(theta_f)

        return backcast, forecast


class NHITSBlock(Block):
    def __init__(self, n_theta_hidden, thetas_dim, pooling_mode, n_pool_kernel_size, backcast_length=10,
                 forecast_length=5, activation='ReLU'):
        self.backcast_length = backcast_length
        self.forecast_length = forecast_length
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

        backcast = F.interpolate(theta_b[:, None, :], size=self.backcast_length,
                                 mode='linear', align_corners=False).squeeze()
        forecast = F.interpolate(theta_f[:, None, :], size=self.forecast_length,
                                 mode='linear', align_corners=False).squeeze()

        return backcast, forecast