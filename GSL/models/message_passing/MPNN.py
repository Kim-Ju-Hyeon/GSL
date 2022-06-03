import torch
from torch_geometric.nn import MessagePassing
import torch.nn as nn
from torch.nn import functional as F
from torch_geometric.utils import degree


class InterCorrealtionStack(MessagePassing):
    def __init__(self, input_dim, hidden_dim, message_norm, GLU=False, single_message=False):
        super().__init__(aggr='add', flow='target_to_source')

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.message_norm = message_norm
        self.GLU = GLU
        self.single_message = single_message

        if self.single_message:
            self.fc_message = nn.Linear(self.hidden_dim[-1], self.hidden_dim[-1])

        else:
            self.fc_message = nn.Linear(self.hidden_dim[-1]*2, self.hidden_dim[-1])

        self.fc_update = nn.Linear(self.hidden_dim[-1]*2, self.hidden_dim[-1])

        self.MLP_stack = nn.ModuleList()
        for i in range(len(self.hidden_dim)):
            if i == 0:
                self.MLP_stack.append(nn.Linear(self.input_dim, self.hidden_dim[i]))
            else:
                self.MLP_stack.append(nn.Linear(self.hidden_dim[i-1], self.hidden_dim[i]))

        if self.GLU:
            if self.single_message:
                self.gated_linear_unit = nn.Linear(self.hidden_dim[-1], self.hidden_dim[-1])
            else:
                self.gated_linear_unit = nn.Linear(self.hidden_dim[-1]*2, self.hidden_dim[-1])

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

        for layer in self.MLP_stack:
            x = layer(x)
            x = F.relu(x)
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
        aggregated_concated_message = torch.cat([x, inputs], dim=-1)
        out = F.relu(self.fc_update(aggregated_concated_message))

        return out
