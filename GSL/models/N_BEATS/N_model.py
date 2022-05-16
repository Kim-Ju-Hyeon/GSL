import torch.nn as nn
from models.N_BEATS.Block import *


class N_model(nn.Module):
    SEASONALITY_BLOCK = 'seasonality'
    TREND_BLOCK = 'trend'
    GENERIC_BLOCK = 'generic'
    N_HITS_BLOCK = 'n_hits'

    def __init__(self, stack_types=(GENERIC_BLOCK, GENERIC_BLOCK, GENERIC_BLOCK),
                 activ='ReLU',
                 num_blocks_per_stack=3,
                 forecast_length=5,
                 backcast_length=10,
                 n_theta_hidden=None,
                 thetas_dim=None,
                 share_weights_in_stack=True,
                 pooling_mode=None,
                 n_pool_kernel_size=None):
        super(N_model, self).__init__()

        if thetas_dim is None:
            thetas_dim = [64, 8]
        if n_theta_hidden is None:
            n_theta_hidden = [32, 32, 32]

        self.activation = activ
        self.stack_types = stack_types
        self.forecast_length = forecast_length
        self.backcast_length = backcast_length
        self.n_theta_hidden = n_theta_hidden
        self.num_blocks_per_stack = num_blocks_per_stack
        self.thetas_dim = thetas_dim
        self.share_weights_in_stack = share_weights_in_stack

        self.pooling_mode = pooling_mode
        self.n_pool_kernel_size = n_pool_kernel_size

        self.stacks = []
        self.parameters = []

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
                    block = block_init(self.n_theta_hidden, self.thetas_dim, self.pooling_mode,
                                       self.n_pool_kernel_size,
                                       self.backcast_length, self.forecast_length, self.activation)
                else:
                    block = block_init(self.n_theta_hidden, self.thetas_dim,
                                       self.backcast_length, self.forecast_length, self.activation)
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

    def forward(self, backcast, edge_index):
        device = backcast.device
        forecast = torch.zeros(size=(backcast.size()[0], self.forecast_length)).to(device=device)

        for stack_id in range(len(self.stacks)):
            for block_id in range(len(self.stacks[stack_id])):
                b, f = self.stacks[stack_id][block_id](backcast, edge_index)
                backcast = backcast.to(device=device) - b
                forecast = forecast.to(device=device) + f
        return backcast, forecast

