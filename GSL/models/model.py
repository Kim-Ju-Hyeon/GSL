from models.GTS.gts_graph_learning2 import GTS_Graph_Learning
from models.GTS.gts_forecasting_module import GTS_Forecasting_Module, GTS_Traffic_Forecasting_Module
from models.GTS.self_attention_graph_learning import Attention_Graph_Learning
from models.MTGNN.mtgnn_graph_learning import MTGNN_Graph_Learning
from utils.adjacency_matrix_sampling import gumbel_softmax_structure_sampling, weight_matrix_construct, top_k_structure_construct


import torch
import torch.nn as nn
from torch.nn import functional as F
from torch_geometric.utils import to_dense_adj, dense_to_sparse


class My_Model(nn.Module):
    def __init__(self, config):
        super(My_Model, self).__init__()

        self.config = config
        self.dataset_conf = config.dataset
        self.node_nums = config.nodes_num

        self.graph_learning_parameter = config.graph_learning
        self.graph_learning_mode = config.graph_learning.mode
        self.sampling_mode = config.graph_learning.sampling
        self.symmetric = config.graph_learning.to_symmetric
        self.graph_learning_sequence = config.graph_learning.sequence

        if self.dataset_conf.name == 'spike_lambda_bin100':
            self.graph_forecasting = GTS_Forecasting_Module(self.config)
        elif (self.dataset_conf.name == 'METR-LA') or (self.dataset_conf.name == 'PEMS-BAY'):
            self.graph_forecasting = GTS_Traffic_Forecasting_Module(self.config)
        else:
            raise ValueError("Non-supported Forecasting Module!")

        if self.graph_learning_mode == 'GTS':
            self.graph_learning = GTS_Graph_Learning(self.config)
        elif self.graph_learning_mode == 'attention':
            self.graph_learning = Attention_Graph_Learning(self.config)
        elif self.graph_learning_mode == 'MTGNN':
            self.graph_learning = MTGNN_Graph_Learning(self.config)
            self.symmetric = False
        elif self.graph_learning_mode == 'GDN':
            pass
        elif self.graph_learning_mode == 'ProbSparse_Attention':
            pass
        else:
            raise ValueError("Invalid graph learning mode")

    def forward(self, inputs, targets, entire_inputs, edge_index):
        batch_size = self.config.train.batch_size

        if self.graph_learning_mode == 'GTS':
            theta = self.graph_learning(entire_inputs, edge_index)
        elif self.graph_learning_mode == 'attention':
            theta = self.graph_learning(entire_inputs)
        elif self.graph_learning_mode == 'MTGNN':
            theta = self.graph_learning()
        else:
            raise ValueError("Invalid graph learning mode")

        if self.sampling_mode == 'Gumbel_softmax':
            batch_edge_index, adj_matrix = gumbel_softmax_structure_sampling(theta, edge_index, batch_size,
                                                                             tau=self.graph_learning_parameter.tau,
                                                                             node_nums=self.node_nums,
                                                                             symmetric=self.symmetric)
            batch_weight_matrix = None

        elif self.sampling_mode == 'Top_k':
            batch_edge_index, adj_matrix = top_k_structure_construct(theta, batch_size,
                                                                     alpha=self.graph_learning_parameter.alpha,
                                                                     k=self.graph_learning_parameter.top_k,
                                                                     node_nums=self.node_nums,
                                                                     symmetric=self.symmetric)
            batch_weight_matrix = None

        elif self.sampling_mode == 'False':
            batch_edge_index, batch_weight_matrix, adj_matrix = weight_matrix_construct(theta, batch_size, self.node_nums, self.symmetric)
        else:
            raise ValueError("Invalid graph sampling mode")

        if (self.graph_learning_mode == 'GTS') and (self.graph_learning_sequence > 1) and (self.sampling_mode == 'Gumbel_softmax'):
            for _ in range(self.graph_learning_sequence-1):
                new_edge_index, _ = dense_to_sparse(adj_matrix)
                theta = self.graph_learning(entire_inputs, new_edge_index)
                batch_edge_index, adj_matrix = gumbel_softmax_structure_sampling(theta, new_edge_index, batch_size,
                                                                                 tau=self.graph_learning_parameter.tau,
                                                                                 node_nums=self.node_nums,
                                                                                 symmetric=self.symmetric
                                                                                 )

        outputs = self.graph_forecasting(inputs, targets, batch_edge_index, weight_matrix=batch_weight_matrix)

        return adj_matrix, outputs