import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np

from models.GTS.graph_learning import GlobalGraphLearning
from models.GTS.DCRNN import DCRNN
from utils.utils import build_edge_idx


class EncoderModel(nn.Module):
    def __init__(self, config):
        super(EncoderModel, self).__init__()
        self.encoder_dcrnn = DCRNN(config)


class DecoderModel(nn.Module):
    def __init__(self, config):
        super(DecoderModel, self).__init__()
        self.decoder_length = config.decoder_length
        self.output_dim = config.output_dim
        self.hidden_dim = config.hidden_dim

        self.decoder_dcrnn = DCRNN(config)
        self.prediction_layer = nn.Linear(self.hidden_dim, self.output_dim)

    def forward(self, inputs, adj, encoder_hidden_state):
        decoder_hidden_state = encoder_hidden_state
        output = inputs

        decoder_hidden_state = self.decoder_dcrnn(output, adj, hidden_state=decoder_hidden_state)
        prediction = self.prediction_layer(decoder_hidden_state[-1].view(-1, self.hidden_dim))

        output = prediction.view(-1, self.num_nodes * self.output_dim)

        return output, decoder_hidden_state


class GTS_Forecasting_Module(nn.Module):
    def __init__(self, config):
        super(GTS_Forecasting_Module, self).__init__()

        self.coonfig = config
        self.device = config.device

        self.nodes_num = config.nodes_num
        self.tau = config.tau
        self.use_teacher_forcing = config.use_teacher_forcing
        self.teacher_forcing_ratio = config.teacher_forcing_ratio

        self.nodes_feas = config.node_features
        self.output_dim = config.output_dim

        self.encoder_length = config.encoder_length
        self.decoder_length = config.decoder_length

        self.encoder_model = EncoderModel(config)
        self.decoder_model = DecoderModel(config)

    def encoder(self, inputs, adj):
        encoder_hidden_state = None
        for t in range(self.encoder_length):
            encoder_hidden_state = self.encoder_model(inputs[t], adj, hidden_state=encoder_hidden_state)

        return encoder_hidden_state

    def decoder(self, targets, encoder_hidden_state, adj):
        # Teacher Forcing 미구현
        outputs = []

        batch_size = encoder_hidden_state.size(0)
        go_symbol = torch.zeros((batch_size, self.nodes_num, self.output_dim), device=self.decive)
        decoder_hidden_state = encoder_hidden_state
        decoder_input = go_symbol

        for t in range(self.decoder_length):
            output, decoder_hidden_state = self.decoder_model(decoder_input, adj, hidden_state=decoder_hidden_state)
            outputs.append(output)

            self.use_teacher_forcing = True if (np.random.random() < self.teacher_forcing_ratio) and \
                                               (self.use_teacher_forcing is True) else False

            if self.training and self.use_teacher_forcing:
                decoder_input = targets[t]
            else:
                decoder_input = output

        outputs = torch.stack(outputs)
        return outputs

    def forward(self, inputs, targets, adj_matrix):
        # DCRNN encoder
        encoder_hidden_state = self.encoder(inputs, adj_matrix)

        # DCRNN decoder
        outputs = self.decoder(targets, encoder_hidden_state, adj_matrix)

        return outputs
