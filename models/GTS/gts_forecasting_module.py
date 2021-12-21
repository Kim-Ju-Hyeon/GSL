import torch
import torch.nn as nn
import numpy as np

from torch.nn import functional as F
from models.GTS.DCRNN import DCRNN
from torch_geometric_temporal.nn.recurrent import DCRNN as tg_dcrnn


class EncoderModel(nn.Module):
    def __init__(self, config):
        super(EncoderModel, self).__init__()

        # self.embedding_layer = nn.Linear(config.dataset.window_size, config.embedding_dim)
        self.num_nodes = config.nodes_num

        self.kernel_size = config.embedding.kernel_size
        self.stride = config.embedding.stride
        self.conv1_dim = config.embedding.conv1_dim
        self.conv2_dim = config.embedding.conv2_dim
        self.fc_dim = config.embedding.fc_dim

        self.nodes_feas = config.node_features

        self.conv1 = torch.nn.Conv1d(self.nodes_feas, self.conv1_dim, self.kernel_size, stride=self.stride)
        self.conv2 = torch.nn.Conv1d(self.conv1_dim, self.conv2_dim, self.kernel_size, stride=self.stride)
        self.fc = torch.nn.Linear(self.fc_dim, 1)

        self.bn1 = torch.nn.BatchNorm1d(self.conv1_dim)
        self.bn2 = torch.nn.BatchNorm1d(self.conv2_dim)

        self.encoder_dcrnn = DCRNN(config)

    def forward(self, inputs, adj, hidden_state=None):
        batch_nodes = inputs.shape[0]
        if len(inputs.shape) == 2:
            inputs = inputs.reshape(batch_nodes, 1, -1)
        x = self.conv1(inputs)
        x = F.relu(x)
        x = self.bn1(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.bn2(x)

        x = x.view(batch_nodes, -1)

        x = self.fc(x)
        x = F.relu(x)

        hidden_state = self.encoder_dcrnn(x, adj, hidden_state)
        return hidden_state


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


class RecurrentGCN(torch.nn.Module):
    def __init__(self, config):
        super(RecurrentGCN, self).__init__()
        self.config = config
        self.num_nodes = config.nodes_num
        self.nodes_feas = config.node_features

        self.kernel_size = config.embedding.kernel_size
        self.stride = config.embedding.stride
        self.conv_dim = config.embedding.conv_dim
        self.embedding_dim = config.embedding.embedding_dim

        self.conv1 = torch.nn.Conv1d(self.nodes_feas, self.conv_dim, self.kernel_size[0],
                                     stride=self.stride[0])
        self.conv2 = torch.nn.Conv1d(self.conv_dim, self.embedding_dim,
                                     self.kernel_size[1], stride=self.stride[1])

        self.bn1 = torch.nn.BatchNorm1d(self.conv_dim)
        self.bn2 = torch.nn.BatchNorm1d(self.embedding_dim)

        self.recurrent = tg_dcrnn(config.embedding.embedding_dim, config.embedding.embedding_dim, 3)

        self.conv3 = torch.nn.Conv1d(1, 1, config.embedding.embedding_dim, stride=1)

        self.out_1 = torch.nn.Linear(self.num_nodes, config.decode_step)
        self.out_2 = torch.nn.Linear(self.num_nodes, config.decode_step)

        if config.decode_dim == 3:
            self.out_3 = torch.nn.Linear(self.num_nodes, config.decode_step)

    def forward(self, x, edge_index, hidden_state):
        batch_nodes = x.shape[0]
        if len(x.shape) == 2:
            x = x.reshape(batch_nodes, 1, -1)
        x = self.conv1(x)
        x = F.relu(x)
        x = self.bn1(x)
        #         print(x.shape)

        x = self.conv2(x)
        x = F.relu(x)
        x = self.bn2(x)
        #         print(x.shape)

        x = x.view(batch_nodes, -1)
        #         print(x.shape)

        h = self.recurrent(x, edge_index, H=hidden_state)
        h = F.relu(h)
        #         print(h.shape)

        h_ = h.view(batch_nodes, 1, self.embedding_dim)
        #         print(h_.shape)
        forecast_h = self.conv3(h_)
        forecast_h = F.relu(forecast_h)
        #         print(forecast_h.shape)

        forecast_h = forecast_h.view(-1)
        #         print(forecast_h.shape)

        out_x = self.out_1(forecast_h)
        out_y = self.out_2(forecast_h)
        output = [out_x, out_y]

        if self.config.decode_dim == 3:
            out_z = self.out_3(forecast_h)
            output = [out_x, out_y, out_z]

        return output, h
