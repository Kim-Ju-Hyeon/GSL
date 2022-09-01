import numpy as np
import torch.nn.functional as F
from models.graph_learning_Attention.probsparseattention import GraphLearningProbSparseAttention
from models.IC_PN_BEATS.Block import *
from models.layer.none_graph_learning_layer import None_Graph_Learning
from models.embedding_module.embed import DataEmbedding

import torch.nn as nn
from torch_geometric.utils import dense_to_sparse

from utils.utils import build_batch_edge_index, build_batch_edge_weight


def attn_to_edge_index(attn):
    _sparse = dense_to_sparse(attn)
    return _sparse[0], _sparse[1]


class IC_PN_BEATS(nn.Module):
    SEASONALITY_BLOCK = 'seasonality'
    TREND_BLOCK = 'trend'
    GENERIC_BLOCK = 'generic'

    def __init__(self, config):
        super(IC_PN_BEATS, self).__init__()
        assert len(config.forecasting_module.n_pool_kernel_size) == len(
            config.forecasting_module.n_stride_size), f'pooling kernel: {len(config.forecasting_module.n_pool_kernel_size)} and stride: ' \
                                                      f'{len(config.forecasting_module.n_stride_size)} is not match '
        assert len(config.forecasting_module.n_pool_kernel_size) == config.forecasting_module.stack_num, 'Pooling num ' \
                                                                                                         'and Stack ' \
                                                                                                         'num is not ' \
                                                                                                         'match '

        self.config = config
        self.nodes_num = config.dataset.nodes_num
        self.num_feature = config.dataset.node_features
        self.n_head = config.graph_learning.n_head
        self.batch_size = config.train.batch_size
        self.embedding_dim = config.forecasting_module.embedding_dim

        if not self.config.dataset.univariate:
            self.embed = DataEmbedding(c_in=1, embedding_dim=self.embedding_dim, batch_size=self.batch_size,
                                       nodes_num=self.nodes_num,
                                       freq=self.config.dataset.freq)

        if self.config.graph_learning.graph_learning:
            self.graph_learning_module = GraphLearningProbSparseAttention(self.config)
        else:
            self.graph_learning_module = None_Graph_Learning(self.config)

        self.attn_matrix = []

        self.activation = config.forecasting_module.activ

        self.stack_num = config.forecasting_module.stack_num
        self.singular_stack_num = config.forecasting_module.singular_stack_num

        self.inter_correlation_block_type = config.forecasting_module.inter_correlation_block_type
        self.forecast_length = config.forecasting_module.forecast_length
        self.backcast_length = config.forecasting_module.backcast_length
        self.n_theta_hidden = config.forecasting_module.n_theta_hidden
        self.thetas_dim = config.forecasting_module.thetas_dim
        self.n_layers = config.forecasting_module.inter_correlation_stack_length
        self.pooling_mode = config.forecasting_module.pooling_mode
        self.n_pool_kernel_size = config.forecasting_module.n_pool_kernel_size
        self.n_stride_size = config.forecasting_module.n_stride_size

        if (self.config.dataset.name == 'METR-LA') or (self.config.dataset.name == 'PEMS-BAY'):
            self.update_only_message = config.forecasting_module.update_only_message
        else:
            self.update_only_message = False

        self.parameters = []

        # Pooling
        self.pooling_stack = []
        for i in range(len(self.n_pool_kernel_size)):
            if self.pooling_mode == 'max':
                pooling_layer = nn.MaxPool1d(kernel_size=self.n_pool_kernel_size[i],
                                             stride=self.n_stride_size[i], ceil_mode=False)
            elif self.pooling_mode == 'average':
                pooling_layer = nn.AvgPool1d(kernel_size=self.n_pool_kernel_size[i],
                                             stride=self.n_stride_size[i], ceil_mode=False)
            else:
                raise ValueError('Invalid Pooling Mode Only "max", "average" is available')

            self.pooling_stack.append(pooling_layer)

        # Make Stack For Trend, Seasonality, Singular
        self.trend_stacks = []
        self.seasonality_stacks = []
        self.sigular_stacks = []
        for stack_id in range(self.stack_num):
            self.trend_stacks.append(self.create_stack('generic'))

        for stack_id in range(self.stack_num):
            self.seasonality_stacks.append(self.create_stack('generic'))

        for stack_id in range(self.singular_stack_num):
            self.sigular_stacks.append(self.create_stack('generic'))

        self.parameters = nn.ParameterList(self.parameters)

    def create_stack(self, stack_type):
        block_init = IC_PN_BEATS.select_block(stack_type)

        if stack_type == IC_PN_BEATS.TREND_BLOCK:
            thetas_dim = [0, 0]

            thetas_dim[0] = 3
            thetas_dim[1] = 3

            block = block_init(inter_correlation_block_type=self.inter_correlation_block_type,
                               n_theta_hidden=self.n_theta_hidden, thetas_dim=thetas_dim,
                               backcast_length=self.backcast_length, forecast_length=self.forecast_length,
                               inter_correlation_stack_length=self.n_layers,
                               update_only_message=self.update_only_message)

        elif stack_type == IC_PN_BEATS.SEASONALITY_BLOCK:
            thetas_dim = [0, 0]

            thetas_dim[0] = 2 * int(self.backcast_length / 2 - 1) + 1
            thetas_dim[1] = 2 * int(self.forecast_length / 2 - 1) + 1

            block = block_init(inter_correlation_block_type=self.inter_correlation_block_type,
                               n_theta_hidden=self.n_theta_hidden, thetas_dim=thetas_dim,
                               backcast_length=self.backcast_length, forecast_length=self.forecast_length,
                               inter_correlation_stack_length=self.n_layers,
                               update_only_message=self.update_only_message)

        elif stack_type == IC_PN_BEATS.GENERIC_BLOCK:
            block = block_init(inter_correlation_block_type=self.inter_correlation_block_type,
                               n_theta_hidden=self.n_theta_hidden, thetas_dim=self.thetas_dim,
                               backcast_length=self.backcast_length, forecast_length=self.forecast_length,
                               inter_correlation_stack_length=self.n_layers,
                               update_only_message=self.update_only_message)

        self.parameters.extend(block.parameters())

        return block

    @staticmethod
    def select_block(block_type):
        if block_type == IC_PN_BEATS.SEASONALITY_BLOCK:
            return Seasonlity_Block
        elif block_type == IC_PN_BEATS.TREND_BLOCK:
            return Trend_Block
        elif block_type == IC_PN_BEATS.GENERIC_BLOCK:
            return Generic_Block
        else:
            raise ValueError("Invalid block type")

    def forward(self, inputs, time_stamp=None, interpretability=False):
        device = inputs.device

        forecast = torch.zeros(size=(inputs.size()[0], self.forecast_length)).to(device=device)
        backcast = torch.zeros(size=(inputs.size()[0], self.backcast_length)).to(device=device)

        _per_trend_backcast = []
        _per_trend_forecast = []
        _per_seasonality_backcast = []
        _per_seasonality_forecast = []
        _singual_backcast = []
        _singual_forecast = []

        _trend_attention_matrix = []
        _seasonality_attention_matrix = []
        _singual_attention_matrix = []

        for stack_index in range(self.stack_num):
            if not self.config.dataset.univariate:
                inputs = self.embed(inputs, time_stamp).squeeze()
            else:
                inputs = inputs.squeeze()

            pooled_inputs = self.pooling_stack[stack_index](inputs)
            trend_input = F.interpolate(pooled_inputs.unsqueeze(dim=1), size=inputs.size()[1],
                                        mode='linear', align_corners=False).squeeze(dim=1)
            seasonality_input = inputs - trend_input

            if self.config.graph_learning.graph_learning:
                trend_attn = self.graph_learning_module(
                    trend_input.view(self.batch_size, self.nodes_num, self.backcast_length))
                trend_batch_edge_index, trend_batch_edge_weight = attn_to_edge_index(trend_attn)

            else:
                edge_index, edge_attr = self.graph_learning_module()
                trend_batch_edge_index = build_batch_edge_index(edge_index, num_graphs=self.batch_size,
                                                                num_nodes=self.nodes_num)
                trend_attn = edge_index

                if edge_attr is None:
                    trend_batch_edge_weight = None
                else:
                    trend_batch_edge_weight = build_batch_edge_weight(edge_attr, num_graphs=self.batch_size)

            trend_b, trend_f = self.trend_stacks[stack_index](trend_input, trend_batch_edge_index,
                                                              trend_batch_edge_weight)

            if self.config.graph_learning.graph_learning:
                seasonality_attn = self.graph_learning_module(
                    seasonality_input.view(self.batch_size, self.nodes_num, self.backcast_length))
                seasonality_batch_edge_index, seasonality_batch_edge_weight = attn_to_edge_index(seasonality_attn)
            else:
                seasonality_attn = trend_attn
                seasonality_batch_edge_index = trend_batch_edge_index
                seasonality_batch_edge_weight = trend_batch_edge_weight
            seasonality_b, seasonality_f = self.seasonality_stacks[stack_index](seasonality_input,
                                                                                seasonality_batch_edge_index,
                                                                                seasonality_batch_edge_weight)

            if interpretability:
                _per_trend_backcast.append(trend_b.cpu().detach().numpy())
                _per_trend_forecast.append(trend_f.cpu().detach().numpy())
                _per_seasonality_backcast.append(seasonality_b.cpu().detach().numpy())
                _per_seasonality_forecast.append(seasonality_f.cpu().detach().numpy())
                _trend_attention_matrix.append(trend_attn.cpu().detach().numpy())
                _seasonality_attention_matrix.append(seasonality_attn.cpu().detach().numpy())

            inputs = inputs - trend_b - seasonality_b

            forecast = forecast + trend_f + seasonality_f
            backcast = backcast + trend_b + seasonality_b

        for singular_stack_index in range(self.singular_stack_num):
            if not self.config.dataset.univariate:
                inputs = self.embed(inputs, time_stamp).squeeze()

            if self.config.graph_learning.graph_learning:
                gl_input = inputs.view(self.batch_size, self.nodes_num, self.backcast_length)
                attn = self.graph_learning_module(gl_input)

                _batch_edge_index, _batch_edge_weight = attn_to_edge_index(attn)

            else:
                edge_index, edge_attr = self.graph_learning_module()
                _batch_edge_index = build_batch_edge_index(edge_index, num_graphs=self.batch_size,
                                                           num_nodes=self.nodes_num)
                attn = edge_index

                if edge_attr is None:
                    _batch_edge_weight = None
                else:
                    _batch_edge_weight = build_batch_edge_weight(edge_attr, num_graphs=self.batch_size)

            singular_b, singular_f = self.sigular_stacks[singular_stack_index](inputs,
                                                                               _batch_edge_index,
                                                                               _batch_edge_weight)

            inputs = inputs - singular_b
            forecast = forecast + singular_f
            backcast = backcast + singular_b

            if interpretability:
                _singual_backcast.append(singular_b.cpu().detach().numpy())
                _singual_forecast.append(singular_f.cpu().detach().numpy())
                _singual_attention_matrix.append(attn.cpu().detach().numpy())

        if interpretability:
            self.per_trend_backcast = np.stack(_per_trend_backcast, axis=0)
            self.per_trend_forecast = np.stack(_per_trend_forecast, axis=0)

            self.per_seasonality_backcast = np.stack(_per_seasonality_backcast, axis=0)
            self.per_seasonality_forecast = np.stack(_per_seasonality_forecast, axis=0)

            self.attn_matrix = {'Trend': _trend_attention_matrix,
                                'Seasonality': _seasonality_attention_matrix,
                                'Singular': _singual_attention_matrix}

            if self.singular_stack_num >= 1:
                self.singual_backcast = np.stack(_singual_backcast, axis=0)
                self.singual_forecast = np.stack(_singual_forecast, axis=0)
            else:
                self.singual_backcast = torch.zeros(1)
                self.singual_forecast = torch.zeros(1)

        return backcast, forecast
