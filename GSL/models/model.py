from collections import defaultdict

import torch

from models.GTS.gts_forecasting_module import GTS_Forecasting_Module, GTS_Traffic_Forecasting_Module
from models.N_BEATS.N_model import N_model
from models.N_BEATS.Parallel_N_model import PN_model

from models.GTS.gts_graph_learning2 import GTS_Graph_Learning2
from models.self_attention_graph_learning import Attention_Graph_Learning
from models.MTGNN.mtgnn_graph_learning import MTGNN_Graph_Learning
from models.GDN.gdn_graph_learning import GDN_Graph_Learning
from models.none_graph_learning import None_Graph_Learning

from utils.adjacency_matrix_sampling import gumbel_softmax_structure_sampling, weight_matrix_construct, \
    top_k_adj_masking_zero, top_k_adj

from utils.utils import build_batch_edge_index, build_batch_edge_weight

import torch.nn as nn
from torch_geometric.utils import to_dense_adj, dense_to_sparse


class My_Model(nn.Module):
    def __init__(self, config):
        super(My_Model, self).__init__()

        self.config = config
        self.device = config.device
        self.dataset_conf = config.dataset
        self.nodes_num = config.dataset.nodes_num
        self.num_feature = config.dataset.node_features

        self.graph_learning_parameter = config.graph_learning
        self.graph_learning_mode = config.graph_learning.mode
        self.sampling_mode = config.graph_learning.sampling
        self.symmetric = config.graph_learning.to_symmetric
        self.graph_learning_sequence = config.graph_learning.sequence
        self.batch_size = config.train.batch_size

        self.adj_matrix = None
        self.attention_matrix = None

        self.get_graph_learning_module()
        self.get_forecasting_module()

        self.preprocess_layer = nn.Linear(self.num_feature, 1)

    def get_graph_learning_module(self):
        if self.graph_learning_mode == 'GTS':
            self.graph_learning = GTS_Graph_Learning2(self.config)
        elif self.graph_learning_mode == 'attention':
            self.graph_learning = Attention_Graph_Learning(self.config)
        elif self.graph_learning_mode == 'MTGNN':
            self.graph_learning = MTGNN_Graph_Learning(self.config)
            self.symmetric = False
        elif self.graph_learning_mode == 'GDN':
            self.graph_learning = GDN_Graph_Learning(self.config)
        elif self.graph_learning_mode == 'ProbSparse_Attention':
            pass
        elif self.graph_learning_mode == 'None':
            self.graph_learning = None_Graph_Learning(self.config)
            self.sampling_mode = 'None'
        else:
            raise ValueError("Invalid graph learning mode")

    def get_forecasting_module(self):
        if self.dataset_conf.name == 'spike_lambda_bin100':
            self.graph_forecasting = GTS_Forecasting_Module(self.config)
        elif self.config.forecasting_module.name == 's2s_dcrnn_traffic':
            self.graph_forecasting = GTS_Traffic_Forecasting_Module(self.config)
        elif self.config.forecasting_module.name == 'n_beats':
            self.graph_forecasting = N_model(self.config.forecasting_module)
        elif self.config.forecasting_module.name == 'pn_beats':
            self.graph_forecasting = PN_model(self.config.forecasting_module)
        else:
            raise ValueError("Non-supported Forecasting Module!")

    def graph_learning_process(self, entire_inputs, edge_index):
        if self.graph_learning_mode == 'GTS':
            theta = self.graph_learning(entire_inputs, edge_index)
            attention_matrix = None
        elif self.graph_learning_mode == 'attention':
            theta, attention_matrix = self.graph_learning(entire_inputs)
            attention_matrix += [theta]
            attention_matrix = torch.stack(attention_matrix, dim=0)
        elif (self.graph_learning_mode == 'MTGNN') or (self.graph_learning_mode == 'GDN'):
            theta = self.graph_learning()
            attention_matrix = [theta]
        elif self.graph_learning_mode == 'None':
            theta, attention_matrix = self.graph_learning()
        else:
            raise ValueError("Invalid graph learning mode")

        if self.sampling_mode == 'Gumbel_softmax':
            batch_edge_index, adj_matrix = gumbel_softmax_structure_sampling(theta, edge_index, self.batch_size,
                                                                             tau=self.graph_learning_parameter.tau,
                                                                             nodes_num=self.nodes_num,
                                                                             symmetric=self.symmetric)
            batch_weight_matrix = None

        elif self.sampling_mode == 'Top_K_Masking':
            batch_edge_index, batch_weight_matrix, adj_matrix = top_k_adj_masking_zero(theta, self.batch_size,
                                                                                       k=self.graph_learning_parameter.top_k,
                                                                                       nodes_num=self.nodes_num,
                                                                                       symmetric=self.symmetric,
                                                                                       device=self.device)
        elif self.sampling_mode == 'Top_K':
            batch_edge_index, adj_matrix = top_k_adj(theta, self.batch_size,
                                                     k=self.graph_learning_parameter.top_k,
                                                     nodes_num=self.nodes_num,
                                                     symmetric=self.symmetric,
                                                     device=self.device)

            batch_weight_matrix = None

        elif self.sampling_mode == 'Weight':
            batch_edge_index, batch_weight_matrix, adj_matrix = weight_matrix_construct(theta, self.batch_size,
                                                                                        self.nodes_num, self.symmetric)

        elif self.sampling_mode == 'None':
            batch_edge_index = build_batch_edge_index(theta, num_graphs=self.batch_size, num_nodes=self.nodes_num)
            adj_matrix = to_dense_adj(theta).squeeze(dim=0)

            if attention_matrix is None:
                batch_weight_matrix = None
            else:
                batch_weight_matrix = build_batch_edge_weight(attention_matrix, num_graphs=self.batch_size)

            attention_matrix = to_dense_adj(theta, edge_attr=attention_matrix).squeeze(dim=0)

        else:
            raise ValueError("Invalid graph sampling mode")

        if (self.graph_learning_mode == 'GTS') and (self.graph_learning_sequence > 1) and (
                self.sampling_mode == 'Gumbel_softmax'):
            for _ in range(self.graph_learning_sequence - 1):
                new_edge_index, _ = dense_to_sparse(adj_matrix)
                theta = self.graph_learning(entire_inputs, new_edge_index)
                batch_edge_index, adj_matrix = gumbel_softmax_structure_sampling(theta, new_edge_index, self.batch_size,
                                                                                 tau=self.graph_learning_parameter.tau,
                                                                                 nodes_num=self.nodes_num,
                                                                                 symmetric=self.symmetric
                                                                                 )
        self.adj_matrix = adj_matrix
        self.attention_matrix = attention_matrix
        return batch_edge_index, batch_weight_matrix

    def forward(self, inputs, targets, entire_inputs, edge_index, interpretability=False):
        batch_edge_index, batch_weight_matrix = self.graph_learning_process(entire_inputs, edge_index)

        if self.config.forecasting_module.name == 's2s_dcrnn_traffic':
            outputs = self.graph_forecasting(inputs, targets, batch_edge_index, weight_matrix=batch_weight_matrix)
        elif (self.config.forecasting_module.name == 'n_beats') or (self.config.forecasting_module.name == 'pn_beats'):
            outputs = defaultdict(list)
            inputs = inputs.permute(0, 2, 1)
            inputs = self.preprocess_layer(inputs)
            backcast, forecast = self.graph_forecasting(inputs.squeeze(), edge_index=batch_edge_index,
                                                        edge_weight=batch_weight_matrix, interpretability=interpretability)
            outputs['backcast'] = backcast
            outputs['forecast'] = forecast

            if interpretability:
                if self.config.forecasting_module.name == 'n_beats':
                    outputs['stack_per_backcast'] = self.graph_forecasting.per_stack_backcast
                    outputs['stack_per_forecast'] = self.graph_forecasting.per_stack_forecast

                    outputs['block_per_backcast'] = self.graph_forecasting.total_backcast_output
                    outputs['block_per_forecast'] = self.graph_forecasting.total_forecast_output

                elif self.config.forecasting_module.name == 'pn_beats':
                    outputs['per_trend_backcast'] = self.graph_forecasting.per_trend_backcast
                    outputs['per_trend_forecast'] = self.graph_forecasting.per_trend_forecast

                    outputs['per_seasonality_backcast'] = self.graph_forecasting.per_seasonality_backcast
                    outputs['per_seasonality_forecast'] = self.graph_forecasting.per_seasonality_forecast

                    outputs['singual_backcast'] = self.graph_forecasting.singual_backcast
                    outputs['singual_forecast'] = self.graph_forecasting.singual_forecast

        else:
            raise ValueError('None supported forecasting module')

        return self.adj_matrix, outputs, self.attention_matrix
