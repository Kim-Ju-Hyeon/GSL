# from torch_geometric_temporal.nn.recurrent import DCRNN as dcrnn
from models.GTS.torch_geometric_temporal_source_code import DCRNN as dcrnn
import torch
import torch.nn as nn

class DCRNN(torch.nn.Module):
    def __init__(self, config):
        super(DCRNN, self).__init__()
        self.num_layer = config.num_layer
        self.node_features = config.node_features
        self.diffusion_k = config.diffusion_k
        self.embedding_dim = config.embedding_dim

        self.recurrent = nn.ModuleList(
            [dcrnn(in_channels=self.embedding_dim, out_channels=self.embedding_dim, K=self.diffusion_k) for _ in range(self.num_layer)]
        )

    def forward(self, x, edge_index, hidden_state=None):
        output = x
        print(x.shape)
        for layer_num, dcgru_layer in enumerate(self.recurrent):
            next_hidden_state = dcgru_layer(output, edge_index, hidden_state)
            output = next_hidden_state

        return output