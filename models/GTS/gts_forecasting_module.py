import torch
import torch.nn as nn
import numpy as np

from torch.nn import functional as F
from models.GTS.DCRNN import DCRNN


class Embedding(nn.Module):
    def __init__(self, config):
        super(Embedding, self).__init__()
        self.nodes_feas = config.node_features
        self.embedding_dim = config.embedding_dim

        self.window_size = config.dataset.window_size
        self.kernel_size = config.forecasting_module.embedding.kernel_size
        self.stride = config.forecasting_module.embedding.stride
        self.conv1_dim = config.forecasting_module.embedding.conv1_dim
        self.conv2_dim = config.forecasting_module.embedding.conv2_dim
        self.fc_dim = config.forecasting_module.embedding.fc_dim

        out_size = 0
        for i in range(len(self.kernel_size)):
            if i == 0:
                out_size = int(((self.window_size - self.kernel_size[i]) / self.stride[i]) + 1)
            else:
                out_size = int(((out_size - self.kernel_size[i]) / self.stride[i]) + 1)
        out_size = out_size * self.conv2_dim

        self.conv1 = torch.nn.Conv1d(self.nodes_feas, self.conv1_dim, self.kernel_size, stride=self.stride)
        self.conv2 = torch.nn.Conv1d(self.conv1_dim, self.conv2_dim, self.kernel_size, stride=self.stride)
        self.fc_conv = torch.nn.Conv1d(self.conv2_dim, 1, 1, stride=1)
        self.fc = torch.nn.Linear(out_size, self.embedding_dim)

        self.bn1 = torch.nn.BatchNorm1d(self.conv1_dim)
        self.bn2 = torch.nn.BatchNorm1d(self.conv2_dim)

    def forward(self, inputs):
        batch_nodes = inputs.shape[0]
        if len(inputs.shape) == 2:
            inputs = inputs.reshape(batch_nodes, 1, -1)

        x = self.conv1(inputs)
        x = F.relu(x)
        x = self.bn1(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.bn2(x)

        x = self.fc_conv(x)
        x = F.relu(x)

        x = self.fc(x)
        return F.relu(x)

class DecoderModel(nn.Module):
    def __init__(self, config):
        super(DecoderModel, self).__init__()
        self.decoder_length = config.decoder_length
        self.output_dim = config.output_dim
        self.hidden_dim = config.hidden_dim

        self.decoder_dcrnn = DCRNN(config)
        self.prediction_layer = nn.Linear(self.hidden_dim, self.output_dim)

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.fill_(0.1)

    def forward(self, inputs, adj, hidden_state):
        decoder_hidden_state = hidden_state
        output = inputs

        decoder_hidden_state = self.decoder_dcrnn(output, adj, hidden_state=decoder_hidden_state)
        prediction = self.prediction_layer(decoder_hidden_state[-1].view(-1, self.hidden_dim))

        output = prediction.view(inputs.shape[0], self.output_dim)

        return output, decoder_hidden_state


class GTS_Forecasting_Module(nn.Module):
    def __init__(self, config):
        super(GTS_Forecasting_Module, self).__init__()

        self.coonfig = config
        self.device = config.device

        self.nodes_num = config.nodes_num
        self.use_teacher_forcing = config.forecasting_module.use_teacher_forcing
        self.teacher_forcing_ratio = config.forecasting_module.teacher_forcing_ratio

        self.nodes_feas = config.node_features
        self.output_dim = config.output_dim

        self.encoder_length = config.encoder_length
        self.decoder_length = config.decoder_length

        self.embedding = Embedding(config)
        self.encoder_model = DCRNN(config)
        self.decoder_model = DCRNN(config)

    def encoder(self, inputs, adj):
        encoder_hidden_state = None
        for t in range(self.encoder_length):
            encoder_hidden_state = self.encoder_model(inputs[:, t].reshape(-1, 1), adj,
                                                      hidden_state=encoder_hidden_state)

        return encoder_hidden_state

    def decoder(self, targets, encoder_hidden_state, adj):
        outputs = []

        go_symbol = torch.zeros((targets.shape[0], 1), device=self.device)
        decoder_hidden_state = encoder_hidden_state
        decoder_input = go_symbol

        for t in range(self.decoder_length):
            output, decoder_hidden_state = self.decoder_model(decoder_input, adj, hidden_state=decoder_hidden_state)
            outputs.append(output)

            self.use_teacher_forcing = True if (np.random.random() < self.teacher_forcing_ratio) and \
                                               (self.use_teacher_forcing is True) else False

            if self.training and self.use_teacher_forcing:
                decoder_input = targets[:, t].reshape(-1, 1)
            else:
                decoder_input = output

        outputs = torch.cat(outputs, dim=-1)
        return outputs

    def forward(self, inputs, targets, adj_matrix):
        # DCRNN encoder
        encoder_hidden_state = self.encoder(inputs, adj_matrix)

        # DCRNN decoder
        outputs = self.decoder(targets, encoder_hidden_state, adj_matrix)

        return outputs
