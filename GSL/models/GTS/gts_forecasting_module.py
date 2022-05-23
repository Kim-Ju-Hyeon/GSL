import torch
import torch.nn as nn

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

        out_size = 0
        for i in range(len(self.kernel_size)):
            if i == 0:
                out_size = int(((self.window_size - self.kernel_size[i]) / self.stride[i]) + 1)
            else:
                out_size = int(((out_size - self.kernel_size[i]) / self.stride[i]) + 1)

        self.conv1 = torch.nn.Conv1d(self.nodes_feas, self.conv1_dim, self.kernel_size[0], stride=self.stride[0])
        self.conv2 = torch.nn.Conv1d(self.conv1_dim, self.conv2_dim, self.kernel_size[1], stride=self.stride[1])
        self.fc_conv = torch.nn.Conv1d(self.conv2_dim, 1, 1, stride=1)
        self.fc = torch.nn.Linear(out_size, self.embedding_dim)

        self.bn1 = torch.nn.BatchNorm1d(self.conv1_dim)
        self.bn2 = torch.nn.BatchNorm1d(self.conv2_dim)

    def forward(self, inputs):
        batch_nodes = inputs.shape[0]
        if len(inputs.shape) == 2:
            inputs = inputs.view(batch_nodes, 1, -1)

        x = self.conv1(inputs)
        x = F.relu(x)
        x = self.bn1(x)

        x = self.conv2(x)
        x = F.relu(x)
        x = self.bn2(x)

        x = self.fc_conv(x)
        x = F.relu(x)

        x = self.fc(x)
        F.relu(x)

        return x.squeeze()


class DecoderModel(nn.Module):
    def __init__(self, config):
        super(DecoderModel, self).__init__()
        self.output_dim = config.dataset.pred_step
        self.hidden_dim = config.hidden_dim

        self.decoder_dcrnn = DCRNN(config, 'Decoder')
        self.prediction_layer = nn.Linear(self.hidden_dim, self.output_dim)

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.fill_(0.1)

    def forward(self, inputs, adj, hidden_state, weight_matrix=None):
        decoder_hidden_state = hidden_state

        decoder_hidden_state = self.decoder_dcrnn(inputs, adj, hidden_state=decoder_hidden_state,
                                                  weight_matrix=weight_matrix)
        prediction = self.prediction_layer(decoder_hidden_state[-1].view(-1, self.hidden_dim))

        output = prediction.view(inputs.shape[0], self.output_dim)

        return output, decoder_hidden_state


class GTS_Forecasting_Module(nn.Module):
    def __init__(self, config):
        super(GTS_Forecasting_Module, self).__init__()

        self.config = config
        self.device = config.device

        self.window_size = config.dataset.window_size
        self.slide = config.dataset.slide

        self.nodes_num = config.nodes_num
        self.use_teacher_forcing = config.forecasting_module.use_teacher_forcing
        self.teacher_forcing_ratio = config.forecasting_module.teacher_forcing_ratio

        self.nodes_feas = config.node_features

        self.encoder_step = config.encoder_step
        self.decoder_step = config.decoder_step

        self.embedding = Embedding(config)
        self.encoder_model = DCRNN(config)
        self.decoder_model = DecoderModel(config)

    def _seq2seq_data_processor(self):
        total_input_size = (self.config.encoder_step + self.decoder_step - 1) * self.slide \
                           + self.window_size

        valid_sampling_locations = []
        valid_sampling_locations += [
            i
            for i in range(0, total_input_size)
            if (i % self.slide) == 0
        ]

        return valid_sampling_locations

    def forward(self, inputs, targets, adj_matrix, weight_matrix=None):
        _input_idx = self._seq2seq_data_processor()

        # DCRNN encoder
        encoder_hidden_state = None
        for i in range(self.encoder_step):
            seq2seq_encoder_input = self.embedding(inputs[:, _input_idx[i]:_input_idx[i] + self.window_size])
            encoder_hidden_state = self.encoder_model(seq2seq_encoder_input, adj_matrix,
                                                      encoder_hidden_state, weight_matrix)

        # DCRNN decoder
        outputs = []

        decoder_hidden_state = encoder_hidden_state
        for j in range(self.decoder_step):
            seq2seq_decoder_input = self.embedding(inputs[:, _input_idx[self.encoder_step + j]:
                                                             _input_idx[self.encoder_step + j] + self.window_size])

            output = self.decoder_model(seq2seq_decoder_input, adj_matrix, decoder_hidden_state, weight_matrix)
            outputs.append(output)

        outputs = torch.cat(outputs, dim=-1)
        return outputs


class GTS_Traffic_Forecasting_Module(nn.Module):
    def __init__(self, config):
        super(GTS_Traffic_Forecasting_Module, self).__init__()

        self.config = config
        self.use_gpu = config.use_gpu
        self.device = config.device

        self.nodes_num = config.nodes_num

        self.nodes_feas = config.node_features

        self.encoder_step = config.encoder_step
        self.decoder_step = config.decoder_step

        self.encoder_model = DCRNN(config, 'Encoder')
        self.decoder_model = DecoderModel(config)

    def forward(self, inputs, targets, adj_matrix, weight_matrix=None):
        # DCRNN encoder
        encoder_hidden_state = None
        for i in range(self.encoder_step):
            encoder_hidden_state = self.encoder_model(inputs[:, :, i], adj_matrix,
                                                      encoder_hidden_state, weight_matrix)

        # DCRNN decoder
        outputs = []
        decoder_input = torch.zeros((self.nodes_num * self.config.train.batch_size, 1))

        if self.use_gpu and (self.device != 'cpu'):
            decoder_input = decoder_input.to(device=self.device)

        decoder_hidden_state = encoder_hidden_state
        for j in range(self.decoder_step):
            output, decoder_hidden_state = self.decoder_model(decoder_input, adj_matrix, decoder_hidden_state,
                                                              weight_matrix)
            outputs.append(output)

            # decoder_input = targets[:, j].unsqueeze(dim=-1) -- Teacher Forcing
            decoder_input = output

        outputs = torch.cat(outputs, dim=-1)
        return outputs
