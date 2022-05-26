import torch
from torch_geometric.nn import MessagePassing
import torch.nn as nn
from torch.nn import functional as F
from torch_geometric.utils import add_self_loops, degree


class InterCorrealtionBlock(MessagePassing):
    def __init__(self, config):
        super().__init__(aggr='add')

        self.input_dim = config.forecasting_module.backcast_length
        self.hidden_dim = config.forecasting_module.n_theta_hidden

        self.MLP_stack = nn.ModuleList()
        for i in range(self.hidden_dim):
            if i ==0:
                self.MLP_stack.append(nn.Linear(self.input_dim, self.hidden_dim[i]))
            else:
                self.MLP_stack.append(nn.Linear(self.hidden_dim[i-1], self.hidden_dim[i]))

    def forward(self, inputs, edge_index, edge_weight):
        pass