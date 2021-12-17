from torch_geometric_temporal.nn.recurrent import DCRNN as dcrnn
import torch
import torch.nn as nn

class DCRNN(torch.nn.Module):
    def __init__(self, config):
        super(DCRNN, self).__init__()
        self.num_layer = config.num_layer
        self.node_features = config.node_features
        self.diffusion_k = config.diffusion_k
        self.hidden_dim = config.hidden_dim

        # self.recurrent = nn.ModuleList(
        #     [dcrnn(in_channels=self.node_features, out_channels=self.hidden_dim,
        #            K=self.diffusion_k) for _ in range(self.num_layer)]
        # )

        self.recurrent = nn.ModuleList()

        for layer_num in range(self.num_layer):
            if layer_num == 0:
                self.recurrent.append(
                    dcrnn(in_channels=self.node_features, out_channels=self.hidden_dim, K=self.diffusion_k))

            else:
                self.recurrent.append(
                    dcrnn(in_channels=self.hidden_dim, out_channels=self.hidden_dim, K=self.diffusion_k))

    def forward(self, x, edge_index, hidden_state=None):
        output = x
        hidden_state_list = []

        for layer_num, dcgru_layer in enumerate(self.recurrent):
            if hidden_state is None:
                next_hidden_state = dcgru_layer(X=output, edge_index=edge_index, H=hidden_state)
            else:
                next_hidden_state = dcgru_layer(X=output, edge_index=edge_index, H=hidden_state[layer_num])

            hidden_state_list.append(next_hidden_state)
            output = next_hidden_state

        return torch.stack(hidden_state_list)