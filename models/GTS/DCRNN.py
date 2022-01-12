from torch_geometric_temporal.nn.recurrent import DCRNN as tgt_dcrnn
import torch
import torch.nn as nn


class DCRNN(torch.nn.Module):
    def __init__(self, config):
        super(DCRNN, self).__init__()
        self.embedding_dim = config.embedding_dim
        self.hidden_dim = config.hidden_dim

        self.num_layer = config.forecasting_module.num_layer
        self.diffusion_k = config.forecasting_module.diffusion_k

        self.recurrent = nn.ModuleList()
        for layer_num in range(self.num_layer):
            if layer_num == 0:
                self.recurrent.append(
                    tgt_dcrnn(in_channels=self.embedding_dim, out_channels=self.hidden_dim, K=self.diffusion_k))
            else:
                self.recurrent.append(
                    tgt_dcrnn(in_channels=self.hidden_dim, out_channels=self.hidden_dim, K=self.diffusion_k))

    def forward(self, x, edge_index, hidden_state=None, weight_matrix=None):
        output = x
        hidden_state_list = []

        for layer_num, dcgru_layer in enumerate(self.recurrent):
            if hidden_state is None:
                next_hidden_state = dcgru_layer(X=output, edge_index=edge_index, edge_weight=weight_matrix, H=hidden_state)
            else:
                next_hidden_state = dcgru_layer(X=output, edge_index=edge_index, edge_weight=weight_matrix, H=hidden_state[layer_num])

            hidden_state_list.append(next_hidden_state)
            output = next_hidden_state

        return torch.stack(hidden_state_list)
