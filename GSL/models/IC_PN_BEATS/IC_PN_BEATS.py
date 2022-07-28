import numpy as np
import torch.nn.functional as F
from models.N_BEATS.Parallel_N_model import PN_model
from models.graph_learning_Attention.probsparseattention import GraphLearningProbSparseAttention
from models.IC_PN_BEATS.Parallel_Block import *

import torch.nn as nn
from torch_geometric.utils import dense_to_sparse


def attn_to_edge_index(attn):
    edge_index = []
    edge_weight = []

    for head in range(attn.shape[0]):
        _sparse = dense_to_sparse(attn)
        edge_index.append(_sparse[0])
        edge_weight.append(_sparse[1])

    edge_index = torch.stack(edge_index, axis=0)
    edge_weight = torch.stack(edge_weight, axis=1)

    return edge_index, edge_weight


class IC_PN_BEATS(PN_model):
    SEASONALITY_BLOCK = 'seasonality'
    TREND_BLOCK = 'trend'
    GENERIC_BLOCK = 'generic'

    def __init__(self, config):
        super(IC_PN_BEATS, self).__init__(config.forecasting_module)
        self.config = config
        self.nodes_num = config.dataset.nodes_num
        self.num_feature = config.dataset.node_features
        self.n_head = config.graph_learning.n_head

        self.preprocess_layer = nn.Linear(self.num_feature, 1)

        self.graph_learning_module = GraphLearningProbSparseAttention(self.config.graph_learning)

        self.attn_matrix = []

    def create_stack(self, stack_type):
        block_init = IC_PN_BEATS.select_block(stack_type)

        if stack_type == IC_PN_BEATS.TREND_BLOCK:
            thetas_dim = [0, 0]

            thetas_dim[0] = 3
            thetas_dim[1] = 3

            block = block_init(inter_correlation_block_type=self.inter_correlation_block_type,
                               n_theta_hidden=self.n_theta_hidden, thetas_dim=thetas_dim,
                               backcast_length=self.backcast_length, forecast_length=self.forecast_length,
                               activation=self.activation,
                               inter_correlation_stack_length=self.n_layers,
                               n_head=self.n_head)

        elif stack_type == IC_PN_BEATS.SEASONALITY_BLOCK:
            thetas_dim = [0, 0]

            thetas_dim[0] = 2 * int(self.backcast_length / 2 - 1) + 1
            thetas_dim[1] = 2 * int(self.forecast_length / 2 - 1) + 1

            block = block_init(inter_correlation_block_type=self.inter_correlation_block_type,
                               n_theta_hidden=self.n_theta_hidden, thetas_dim=thetas_dim,
                               backcast_length=self.backcast_length, forecast_length=self.forecast_length,
                               activation=self.activation,
                               inter_correlation_stack_length=self.n_layers,
                               n_head=self.n_head)

        elif stack_type == IC_PN_BEATS.GENERIC_BLOCK:
            block = block_init(inter_correlation_block_type=self.inter_correlation_block_type,
                               n_theta_hidden=self.n_theta_hidden, thetas_dim=self.thetas_dim,
                               backcast_length=self.backcast_length, forecast_length=self.forecast_length,
                               activation=self.activation,
                               inter_correlation_stack_length=self.n_layers,
                               n_head=self.n_head)

        self.parameters.extend(block.parameters())

        return block


    @staticmethod
    def select_block(block_type):
        if block_type == IC_PN_BEATS.SEASONALITY_BLOCK:
            return Parallel_SeasonalityBlock
        elif block_type == IC_PN_BEATS.TREND_BLOCK:
            return Parallel_TrendBlock
        elif block_type == IC_PN_BEATS.GENERIC_BLOCK:
            return Parallel_GenericBlock
        else:
            raise ValueError("Invalid block type")

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

        _attention_matrix = []

        for stack_index in range(self.stack_num):
            gl_input = inputs.view(self.batch_size, self.nodes_num, self.backcast_length)
            attn = self.graph_learning_module(gl_input, gl_input)

            multi_head_batch_edge_index, multi_head_batch_edge_weight = attn_to_edge_index(attn)

            pooled_inputs = self.pooling_stack[stack_index](inputs.unsqueeze(dim=1))
            trend_input = F.interpolate(pooled_inputs, size=inputs.size()[1],
                                        mode='linear', align_corners=False).squeeze(dim=1)
            seasonality_input = inputs - trend_input

            trend_b, trend_f = self.trend_stacks[stack_index](pooled_inputs, multi_head_batch_edge_index,
                                                              multi_head_batch_edge_weight)

            seasonality_b, seasonality_f = self.seasonality_stacks[stack_index](seasonality_input,
                                                                                multi_head_batch_edge_index,
                                                                                multi_head_batch_edge_weight)

            if interpretability:
                _per_trend_backcast.append(trend_b.cpu().numpy())
                _per_trend_forecast.append(trend_f.cpu().numpy())
                _per_seasonality_backcast.append(seasonality_b.cpu().numpy())
                _per_seasonality_forecast.append(seasonality_f.cpu().numpy())
                _attention_matrix.append(attn.cpu().numpy())

            inputs = inputs - trend_b - seasonality_b

            forecast = forecast + trend_f + seasonality_f
            backcast = backcast + trend_b + seasonality_b

        for singular_stack_index in range(self.singular_stack_num):
            gl_input = inputs.view(self.batch_size, self.nodes_num, self.backcast_length)
            attn = self.graph_learning_module(gl_input, gl_input)

            multi_head_batch_edge_index, multi_head_batch_edge_weight = attn_to_edge_index(attn)

            singular_b, singular_f = self.sigular_stacks[singular_stack_index](inputs,
                                                                               multi_head_batch_edge_index,
                                                                               multi_head_batch_edge_weight)

            if interpretability:
                _singual_backcast.append(singular_b.cpu().numpy())
                _singual_forecast.append(singular_f.cpu().numpy())
                _attention_matrix.append(attn.cpu().numpy())

            inputs = inputs - singular_b

            forecast = forecast + singular_f
            backcast = backcast + singular_b

        if interpretability:
            self.per_trend_backcast.append(np.stack(_per_trend_backcast, axis=0))
            self.per_trend_forecast.append(np.stack(_per_trend_forecast, axis=0))

            self.per_seasonality_backcast.append(np.stack(_per_seasonality_backcast, axis=0))
            self.per_seasonality_forecast.append(np.stack(_per_seasonality_forecast, axis=0))

            self.attn_matrix.append(np.stack(_attention_matrix, axis=0))

            if self.singular_stack_num >= 1:
                self.singual_backcast.append(np.stack(_singual_backcast, axis=0))
                self.singual_forecast.append(np.stack(_singual_forecast, axis=0))
            else:
                self.singual_backcast = torch.zeros(1)
                self.singual_forecast = torch.zeros(1)

        return backcast, forecast
