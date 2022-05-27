import torch
from torch_geometric.nn import MessagePassing
import torch.nn as nn
from torch.nn import functional as F
from torch_geometric.utils import add_self_loops, degree


class InterCorrealtionStack(MessagePassing):
    def __init__(self, config):
        super().__init__(aggr='add', flow='source_to_target')

        self.input_dim = config.forecasting_module.backcast_length
        self.hidden_dim = config.forecasting_module.n_theta_hidden

        self.fc_cat = nn.Linear(self.hidden_dim[-1]*2, self.hidden_dim[-1])
        self.fc_out = nn.Linear(self.hidden_dim[-1], self.hidden_dim[-1])

        self.MLP_stack = nn.ModuleList()
        for i in range(len(self.hidden_dim)):
            if i ==0:
                self.MLP_stack.append(nn.Linear(self.input_dim, self.hidden_dim[i]))
            else:
                self.MLP_stack.append(nn.Linear(self.hidden_dim[i-1], self.hidden_dim[i]))

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.fill_(0.1)

    def forward(self, inputs, edge_index, edge_weight=None):
        x = inputs
        for layer in self.MLP_stack:
            x = layer(x)

        self.propagate(edge_index=edge_index)

    def message(self, x_i, x_j):
        x = torch.cat([x_i, x_j], dim=-1)
        x = F.relu(self.fc_cat(x))
        return x

    def update(self, inputs):
        return F.relu(self.fc_out(inputs))
