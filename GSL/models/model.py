from models.GTS.gts_graph_learning2 import GTS_Graph_Learning2
from models.GTS.gts_forecasting_module import GTS_Forecasting_Module, GTS_Traffic_Forecasting_Module
from models.GTS.self_attention_graph_learning import Attention_Graph_Learning
from models.MTGNN.mtgnn_graph_learning import MTGNN_Graph_Learning
from models.GDN.gdn_graph_learning import GDN_Graph_Learning
from utils.adjacency_matrix_sampling import gumbel_softmax_structure_sampling, weight_matrix_construct, \
    top_k_adj_masking_zero, top_k_adj

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch_geometric.utils import to_dense_adj, dense_to_sparse


class My_Model(nn.Module):
    def __init__(self, config):
        super(My_Model, self).__init__()

        self.config = config
        self.device = config.device
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
        else:
            raise ValueError("Invalid graph learning mode")

    def forward(self, inputs, targets, entire_inputs, edge_index):
        batch_size = self.config.train.batch_size

        if self.graph_learning_mode == 'GTS':
            theta = self.graph_learning(entire_inputs, edge_index)
            attention_matrix = None
        elif self.graph_learning_mode == 'attention':
            theta, attention_matrix = self.graph_learning(entire_inputs)
        elif (self.graph_learning_mode == 'MTGNN') or (self.graph_learning_mode == 'GDN'):
            theta = self.graph_learning()
            attention_matrix = [theta]
        else:
            raise ValueError("Invalid graph learning mode")

        if self.sampling_mode == 'Gumbel_softmax':
            batch_edge_index, adj_matrix = gumbel_softmax_structure_sampling(theta, edge_index, batch_size,
                                                                             tau=self.graph_learning_parameter.tau,
                                                                             node_nums=self.node_nums,
                                                                             symmetric=self.symmetric)
            batch_weight_matrix = None

        elif self.sampling_mode == 'Top_k_Masking':
            batch_edge_index, batch_weight_matrix, adj_matrix = top_k_adj_masking_zero(theta, batch_size,
                                                                                       k=self.graph_learning_parameter.top_k,
                                                                                       node_nums=self.node_nums,
                                                                                       symmetric=self.symmetric,
                                                                                       device=self.device)
        elif self.sampling_mode == 'Top_K':
            batch_edge_index, adj_matrix = top_k_adj(theta, batch_size,
                                                     k=self.graph_learning_parameter.top_k,
                                                     node_nums=self.node_nums,
                                                     symmetric=self.symmetric,
                                                     device=self.device)

            batch_weight_matrix = None

        elif self.sampling_mode == 'None':
            batch_edge_index, batch_weight_matrix, adj_matrix = weight_matrix_construct(theta, batch_size,
                                                                                        self.node_nums, self.symmetric)
        else:
            raise ValueError("Invalid graph sampling mode")

        if (self.graph_learning_mode == 'GTS') and (self.graph_learning_sequence > 1) and (
                self.sampling_mode == 'Gumbel_softmax'):
            for _ in range(self.graph_learning_sequence - 1):
                new_edge_index, _ = dense_to_sparse(adj_matrix)
                theta = self.graph_learning(entire_inputs, new_edge_index)
                batch_edge_index, adj_matrix = gumbel_softmax_structure_sampling(theta, new_edge_index, batch_size,
                                                                                 tau=self.graph_learning_parameter.tau,
                                                                                 node_nums=self.node_nums,
                                                                                 symmetric=self.symmetric
                                                                                 )

        outputs = self.graph_forecasting(inputs, targets, batch_edge_index, weight_matrix=batch_weight_matrix)

        return adj_matrix, outputs, attention_matrix
