from collections import defaultdict

import torch
import torch.nn.functional as F
from models.N_BEATS.Parallel_N_model import PN_model
from models.graph_learning_Attention.probsparseattention import GraphLearningProbSparseAttention

from utils.utils import build_batch_edge_index, build_batch_edge_weight

import torch.nn as nn
from torch_geometric.utils import to_dense_adj, dense_to_sparse


class IC_PN_BEATS(PN_model):
    def __init__(self, config):
        super(IC_PN_BEATS, self).__init__(config.forecasting_module)
        self.config = config
        self.nodes_num = config.dataset.nodes_num
        self.num_feature = config.dataset.node_features

        self.preprocess_layer = nn.Linear(self.num_feature, 1)

        self.graph_learning_module = GraphLearningProbSparseAttention(self.config.graph_learning)
        self.attn_matrix = []
        self.adj_matrix = []

    def forward(self, inputs, interpretability=False):
        device = inputs.device
        inputs = inputs.permute(0, 2, 1)
        inputs = self.preprocess_layer(inputs).squeeze()

        forecast = torch.zeros(size=(inputs.size()[0], self.forecast_length)).to(device=device)
        backcast = torch.zeros(size=(inputs.size()[0], self.backcast_length)).to(device=device)

        _per_trend_backcast = []
        _per_trend_forecast = []
        _per_seasonality_backcast = []
        _per_seasonality_forecast = []
        _singual_backcast = []
        _singual_forecast = []

        for stack_index in range(self.stack_num):
            gl_input = inputs.view(self.batch_size, self.nodes_num, self.backcast_length)
            attn = self.graph_learning_module(gl_input, gl_input)

            pooled_inputs = self.pooling_stack[stack_index](inputs.unsqueeze(dim=1))
            trend_input = F.interpolate(pooled_inputs, size=inputs.size()[1],
                                        mode='linear', align_corners=False).squeeze(dim=1)
            seasonality_input = inputs - trend_input




