import torch.nn as nn
from models.N_BEATS.Block import *


class N_model(nn.Module):
    SEASONALITY_BLOCK = 'seasonality'
    TREND_BLOCK = 'trend'
    GENERIC_BLOCK = 'generic'
    N_HITS_BLOCK = 'n_hits'

    def __init__(self, config):
        super(N_model, self).__init__()
        self.activation = config.activ
        self.stack_types = config.stack_types
        self.inter_correlation_block_type = config.inter_correlation_block_type
        self.forecast_length = config.forecast_length
        self.backcast_length = config.backcast_length
        self.n_theta_hidden = config.n_theta_hidden
        self.num_blocks_per_stack = config.num_blocks_per_stack
        self.thetas_dim = config.thetas_dim
        self.share_weights_in_stack = config.share_weights_in_stack

        self.pooling_mode = config.pooling_mode
        self.n_pool_kernel_size = config.n_pool_kernel_size

        self.stacks = []
        self.parameters = []

        self.per_stack_backcast = []
        self.per_stack_forecast = []

        for stack_id in range(len(self.stack_types)):
            self.stacks.append(self.create_stack(stack_id))

        self.parameters = nn.ParameterList(self.parameters)

    def create_stack(self, stack_id):
        stack_type = self.stack_types[stack_id]

        blocks = []
        for block_id in range(self.num_blocks_per_stack):
            block_init = N_model.select_block(stack_type)
            if self.share_weights_in_stack and block_id != 0:
                block = blocks[-1]
            else:
                if stack_type == N_model.N_HITS_BLOCK:
                    block = block_init(inter_correlation_block_type=self.inter_correlation_block_type,
                                       n_theta_hidden=self.n_theta_hidden,
                                       thetas_dim=self.thetas_dim, pooling_mode=self.pooling_mode,
                                       n_pool_kernel_size=self.n_pool_kernel_size,
                                       backcast_length=self.backcast_length,
                                       forecast_length=self.forecast_length,
                                       activation=self.activation)

                elif stack_type == N_model.TREND_BLOCK:
                    self.thetas_dim[0], self.thetas_dim[1] = 3, 3
                    block = block_init(inter_correlation_block_type=self.inter_correlation_block_type,
                                       n_theta_hidden=self.n_theta_hidden, thetas_dim=self.thetas_dim,
                                       backcast_length=self.backcast_length, forecast_length=self.forecast_length,
                                       activation=self.activation)
                elif stack_type == N_model.SEASONALITY_BLOCK:
                    self.thetas_dim[0] = 2 * int(self.backcast_length / 2 - 1) + 1
                    self.thetas_dim[1] = 2 * int(self.forecast_length / 2 - 1) + 1

                    block = block_init(inter_correlation_block_type=self.inter_correlation_block_type,
                                       n_theta_hidden=self.n_theta_hidden, thetas_dim=self.thetas_dim,
                                       backcast_length=self.backcast_length, forecast_length=self.forecast_length,
                                       activation=self.activation)
                else:
                    block = block_init(inter_correlation_block_type=self.inter_correlation_block_type,
                                       n_theta_hidden=self.n_theta_hidden, thetas_dim=self.thetas_dim,
                                       backcast_length=self.backcast_length, forecast_length=self.forecast_length,
                                       activation=self.activation)
                self.parameters.extend(block.parameters())
            blocks.append(block)
        return blocks

    @staticmethod
    def select_block(block_type):
        if block_type == N_model.SEASONALITY_BLOCK:
            return GNN_SeasonalityBlock
        elif block_type == N_model.TREND_BLOCK:
            return GNN_TrendBlock
        elif block_type == N_model.GENERIC_BLOCK:
            return GNN_GenericBlock
        elif block_type == N_model.N_HITS_BLOCK:
            return GNN_NHITSBlock
        else:
            raise ValueError("Invalid block type")

    def forward(self, backcast, edge_index, edge_weight=None, interpretability=False):
        device = backcast.device
        forecast = torch.zeros(size=(backcast.size()[0], self.forecast_length)).to(device=device)
        sum_of_backcast = torch.zeros(size=(backcast.size()[0], self.backcast_length)).to(device=device)
        self.per_stack_backcast = []
        self.per_stack_forecast = []
        for stack_id in range(len(self.stacks)):
            stacks_forecast = torch.zeros(size=(backcast.size()[0], self.forecast_length)).to(device=device)
            for block_id in range(len(self.stacks[stack_id])):
                b, f = self.stacks[stack_id][block_id](backcast, edge_index, edge_weight)
                backcast = backcast.to(device=device) - b
                forecast = forecast.to(device=device) + f
                sum_of_backcast += b

                stacks_forecast += f
            if interpretability:
                self.per_stack_backcast.append(backcast.cpu())
                self.per_stack_forecast.append(stacks_forecast.cpu())
        return sum_of_backcast, forecast
