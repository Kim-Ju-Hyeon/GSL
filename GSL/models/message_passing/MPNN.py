import torch
from torch_geometric.nn import MessagePassing
import torch.nn as nn
from torch.nn import functional as F
from torch_geometric.utils import degree


class InterCorrealtionStack(MessagePassing):
    def __init__(self, hidden_dim, message_norm, GLU=False, single_message=False, update_only_message=False, none_gnn=False):
        super().__init__(aggr='add', flow='target_to_source')
        self.hidden_dim = hidden_dim
        self.message_norm = message_norm
        self.GLU = GLU
        self.single_message = single_message
        self.update_only_message = update_only_message
        self.none_gnn = none_gnn

        if self.single_message:
            self.fc_message = nn.Linear(self.hidden_dim, self.hidden_dim)
        else:
            self.fc_message = nn.Linear(self.hidden_dim*2, self.hidden_dim)

        if self.GLU:
            if self.single_message:
                self.gated_linear_unit = nn.Linear(self.hidden_dim, self.hidden_dim)
            else:
                self.gated_linear_unit = nn.Linear(self.hidden_dim*2, self.hidden_dim)

        if self.update_only_message:
            self.fc_update = nn.Linear(self.hidden_dim, self.hidden_dim)
        else:
            self.fc_update = nn.Linear(self.hidden_dim * 2, self.hidden_dim)

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.fill_(0.1)

    def forward(self, inputs, edge_index, edge_weight=None):
        x = inputs

        if self.message_norm:
            row, col = edge_index
            deg = degree(col, x.size(0), dtype=x.dtype)
            deg_inv_sqrt = deg.pow(-0.5)
            deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
            norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        else:
            norm = None

        return self.propagate(edge_index=edge_index, x=x, edge_weight=edge_weight, norm=norm)

    def message(self, x_i, x_j):
        if self.single_message:
            x = x_j
        else:
            x = torch.cat([x_i, x_j], dim=-1)
        message = F.relu(self.fc_message(x))
        if self.GLU:
            x = self.gated_linear_unit(x)
            gate = torch.sigmoid(x)
            message = torch.mul(gate, message)

        return message

    def update(self, inputs, x):
        if self.update_only_message:
            aggregated_concated_message = inputs
        else:
            aggregated_concated_message = torch.cat([x, inputs], dim=-1)

        if self.none_gnn:
            return inputs

        out = F.relu(self.fc_update(aggregated_concated_message))

        return out
