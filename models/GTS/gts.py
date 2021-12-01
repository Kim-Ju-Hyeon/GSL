import torch
import torch.nn as nn
from torch.nn import functional as F

from models.GTS.graph_learning import GlobalGraphLearning
from models.GTS.DCRNN import DCRNN
from utils.utils import build_edge_idx

class GTSModel(nn.Module):
    def __init__(self, config):
        super().__init()
        self.coonfig = config

        self.nodes_num = config.nodes_num
        self.tau = config.tau

        self.nodes_feas = config.node_features
        self.output_dim = config.output_dim

        self.encoder_length = config.encoder_length
        self.decoder_length = config.decoder_length

        self.graph_learning = GlobalGraphLearning(config)

        self.encoder_dcrnn = DCRNN(config)
        self.decoder_dcrnn = DCRNN(config)

    def encoder(self, inputs, adj):
        encoder_hidden_state = None
        for t in range(self.encoder_length):
            encoder_hidden_state = self.encoder_dcrnn(inputs[t], adj, encoder_hidden_state)

        return encoder_hidden_state

    def decoder(self, encoder_hidden_state, inputs, adj):
        # Teaching Force 미구현

        outputs = []

        batch_size = encoder_hidden_state.size(0)
        go_symbol = torch.zeros((batch_size, self.nodes_num, self.output_dim))
        decoder_hidden_state = encoder_hidden_state
        decoder_input = go_symbol

        for t in range(self.decoder_length):
            decoder_hidden_state = self.decoder_model(decoder_input, adj, decoder_hidden_state)
            outputs.append(decoder_hidden_state)

        outputs = torch.stack(outputs)
        return outputs

    def forward(self, inputs, entire_inputs):
        # graph structure learning
        edge_index = build_edge_idx(num_nodes=self.nodes_num)

        z = self.graph_learning(entire_inputs, edge_index)
        z_1 = F.gumbel_softmax(z, tau=self.tau, hard=True)
        z_1 = torch.transpose(z_1, 0, 1)

        edge_ = []

        for ii, rel in enumerate(z_1[0]):
            if bool(rel):
                edge_.append(edge_index[:, ii])

        adj_matrix = torch.stack(edge_, dim=-1)

        # DCRNN encoder
        encoder_hidden_state = self.encoder(inputs, adj_matrix)

        # DCRNN decoder
        outputs = self.decoder(encoder_hidden_state, adj_matrix)




