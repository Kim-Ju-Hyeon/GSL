from torch_geometric_temporal.nn.recurrent import DCRNN as dcrnn
import torch
import torch.nn as nn

class DCRNN(torch.nn.Module):
    def __init__(self, config):
        super(DCRNN, self).__init__()
        self.num_layer = config.dcrnn_num_layer
        self.node_features = config.node_features
        self.diffusion_k = config.diffusion_k

        self.recurrent = nn.ModuleList(
            [dcrnn(self.node_features, self.node_features, self.diffusion_k) for _ in range(self.num_layer)]
        )

    def forward(self, x, edge_index):
        output = x
        for layer_num, dcgru_layer in enumerate(self.dcgru_layers):
            next_hidden_state = dcgru_layer(output, edge_index)
            output = next_hidden_state

        return output