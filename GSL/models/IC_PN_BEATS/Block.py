import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.utils import squeeze_dim
from models.message_passing.MPNN import InterCorrealtionStack
from torch_geometric.nn import GATConv


class TrendGenerator(nn.Module):
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


class SeasonalityGenerator(nn.Module):
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


class GNN_Block(nn.Module):
    def __init__(self, inter_correlation_block_type, n_theta_hidden, thetas_dim, backcast_length=10, forecast_length=5,
                 inter_correlation_stack_length=1, update_only_message=False):

        super().__init__()
        self.update_only_message = update_only_message
        self.inter_correlation_block_type = inter_correlation_block_type
        self.n_theta_hidden = n_theta_hidden
        self.thetas_dim = thetas_dim
        self.n_layers = inter_correlation_stack_length

        self.backcast_length = backcast_length
        self.forecast_length = forecast_length

        self.MLP_stack = nn.ModuleList()
        for i in range(len(self.n_theta_hidden)):
            if i == 0:
                self.MLP_stack.append(nn.Linear(self.backcast_length, self.n_theta_hidden[i]))
                self.MLP_stack.append(nn.ReLU())
                # self.MLP_stack.append(nn.LayerNorm(self.n_theta_hidden[i]))
            else:
                self.MLP_stack.append(nn.Linear(self.n_theta_hidden[i - 1], self.n_theta_hidden[i]))
                self.MLP_stack.append(nn.ReLU())
                # self.MLP_stack.append(nn.LayerNorm(self.n_theta_hidden[i]))

        self.Inter_Correlation_Block = nn.ModuleList()
        for _ in range(self.n_layers):
            if self.inter_correlation_block_type == 'MPNN':
                self.Inter_Correlation_Block.append(InterCorrealtionStack(
                    hidden_dim=self.n_theta_hidden[-1],
                    message_norm=True,
                    update_only_message=self.update_only_message))

            elif self.inter_correlation_block_type == 'MPGLU':
                self.Inter_Correlation_Block.append(InterCorrealtionStack(
                    hidden_dim=self.n_theta_hidden[-1],
                    message_norm=True,
                    GLU=True,
                    update_only_message=self.update_only_message))

            elif self.inter_correlation_block_type == 'MP_single_message':
                self.Inter_Correlation_Block.append(InterCorrealtionStack(
                    hidden_dim=self.n_theta_hidden[-1],
                    message_norm=True,
                    single_message=True,
                    update_only_message=self.update_only_message))

            elif self.inter_correlation_block_type == 'MPGLU_single_message':
                self.Inter_Correlation_Block.append(InterCorrealtionStack(
                    hidden_dim=self.n_theta_hidden[-1],
                    message_norm=True,
                    GLU=True,
                    single_message=True,
                    update_only_message=self.update_only_message))

            elif self.inter_correlation_block_type == 'GAT':
                self.Inter_Correlation_Block.append(GATConv(
                    in_channels=self.n_theta_hidden[-1],
                    out_channels=self.n_theta_hidden[-1],
                    heads=4,
                    concat=False
                ))

            elif self.inter_correlation_block_type == 'None_GNN':
                self.Inter_Correlation_Block.append(InterCorrealtionStack(
                    hidden_dim=self.n_theta_hidden[-1],
                    message_norm=False,
                    GLU=False,
                    single_message=True,
                    update_only_message=self.update_only_message,
                    none_gnn=True))

            else:
                raise ValueError('Invalid Inter Correlation Block')

        # self.norm_layer = nn.ModuleList()
        # for i in range(self.n_layers):
        #     self.norm_layer.append(nn.LayerNorm(self.n_theta_hidden[-1]))

        # self.drop_out = nn.Dropout(p=0.5)
        self.theta_b_fc = nn.Linear(n_theta_hidden[-1], thetas_dim[0], bias=False)
        self.theta_f_fc = nn.Linear(n_theta_hidden[-1], thetas_dim[1], bias=False)

    def forward(self, x, edge_index, edge_weight=None):
        x = squeeze_dim(x)

        for mlp in self.MLP_stack:
            x = mlp(x)
            # x = self.drop_out(x)

        for ii, layer in enumerate(self.Inter_Correlation_Block):
            if self.inter_correlation_block_type == 'GAT':
                x = layer(x, edge_index)

            else:
                x = layer(x, edge_index, edge_weight)
            x = F.relu(x)
            # x = self.norm_layer[ii](x)
            # x = self.drop_out(x)

        return x


class Trend_Block(GNN_Block):
    def __init__(self,
                 inter_correlation_block_type,
                 n_theta_hidden,
                 thetas_dim,
                 backcast_length=10,
                 forecast_length=5,
                 inter_correlation_stack_length=1,
                 update_only_message=False):
        super().__init__(inter_correlation_block_type, n_theta_hidden,
                         thetas_dim, backcast_length, forecast_length,
                         inter_correlation_stack_length, update_only_message)

        self.backcast_trend_model = TrendGenerator(thetas_dim[0], backcast_length)
        self.forecast_trend_model = TrendGenerator(thetas_dim[1], forecast_length)

    def forward(self, x, edge_index, edge_weight):
        x = super().forward(x, edge_index, edge_weight)

        backcast = self.backcast_trend_model(self.theta_b_fc(x))
        forecast = self.forecast_trend_model(self.theta_f_fc(x))

        return backcast, forecast


class Seasonlity_Block(GNN_Block):
    def __init__(self,
                 inter_correlation_block_type,
                 n_theta_hidden,
                 thetas_dim,
                 backcast_length=10,
                 forecast_length=5,
                 inter_correlation_stack_length=1,
                 update_only_message=False):
        super().__init__(inter_correlation_block_type, n_theta_hidden,
                         thetas_dim, backcast_length, forecast_length,
                         inter_correlation_stack_length, update_only_message)

        self.backcast_seasonality_model = SeasonalityGenerator(backcast_length)
        self.forecast_seasonality_model = SeasonalityGenerator(forecast_length)

    def forward(self, x, edge_index, edge_weight):
        x = super().forward(x, edge_index, edge_weight)

        backcast = self.backcast_seasonality_model(self.theta_b_fc(x))
        forecast = self.forecast_seasonality_model(self.theta_f_fc(x))

        return backcast, forecast


class Generic_Block(GNN_Block):
    def __init__(self,
                 inter_correlation_block_type,
                 n_theta_hidden,
                 thetas_dim,
                 backcast_length=10,
                 forecast_length=5,
                 inter_correlation_stack_length=1,
                 update_only_message=False):
        super().__init__(inter_correlation_block_type, n_theta_hidden,
                         thetas_dim, backcast_length, forecast_length,
                         inter_correlation_stack_length, update_only_message)

        self.backcast_fc = nn.Linear(thetas_dim[0], backcast_length)
        self.forecast_fc = nn.Linear(thetas_dim[1], forecast_length)

    def forward(self, x, edge_index, edge_weight=None):
        x = super().forward(x, edge_index, edge_weight)

        theta_b = self.theta_b_fc(x)
        theta_f = self.theta_f_fc(x)

        backcast = self.backcast_fc(theta_b)
        forecast = self.forecast_fc(theta_f)

        return backcast, forecast
