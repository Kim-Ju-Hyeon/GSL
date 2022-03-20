from models.GTS.gts_graph_learning import GTS_Graph_Learning
from models.GTS.gts_forecasting_module import GTS_Forecasting_Module, GTS_Traffic_Forecasting_Module
from models.GTS.self_attention_graph_learning import Attention_Graph_Learning
from utils.utils import build_batch_edge_index, build_batch_edge_weight

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch_geometric.utils import to_undirected, to_dense_adj, dense_to_sparse


class GTS_Model(nn.Module):
    def __init__(self, config):
        super(GTS_Model, self).__init__()

        self.config = config
        self.dataset_conf = config.dataset

        self.node_nums = config.nodes_num
        self.tau = config.tau

        self.undirected_adj = config.graph_learning.to_symmetric

        self.graph_learning_mode = config.graph_learning.mode
        self.graph_learning_sequence = config.graph_learning.sequence

        if self.dataset_conf.name == 'spike_lambda_bin100':
            self.graph_forecasting = GTS_Forecasting_Module(self.config)
        elif self.dataset_conf.name == 'METR-LA':
            self.graph_forecasting = GTS_Traffic_Forecasting_Module(self.config)

        elif self.dataset_conf.name == 'PEMS-BAY':
            self.graph_forecasting = GTS_Traffic_Forecasting_Module(self.config)
        else:
            raise ValueError("Non-supported dataset!")



        if self.graph_learning_mode == 'weight':
            self.graph_learning = GTS_Graph_Learning(self.config, 1)

        elif self.graph_learning_mode == 'adj':
            self.graph_learning = GTS_Graph_Learning(self.config, 2)

        elif self.graph_learning_mode == 'both':
            self.graph_learning = GTS_Graph_Learning(self.config, 2)
            self.weight_learning = GTS_Graph_Learning(self.config, 1)

        elif self.graph_learning_mode == 'attention':
            self.graph_learning = Attention_Graph_Learning(self.config)

        else:
            raise ValueError("Invalid graph learning mode")

        # self.correlation_softmax = nn.Softmax(dim=1)
        self.correlation_act = nn.Sigmoid()

    def _gumbel_softmax_structure_sampling(self, adj, init_edge_index, batch_size):
        edge_probability = F.gumbel_softmax(adj, tau=self.tau, hard=True)
        connect = torch.where(edge_probability[:, 0])

        adj_matrix = torch.stack([init_edge_index[0, :][connect],
                                  init_edge_index[1, :][connect]])
        if self.undirected_adj:
            adj_matrix = to_undirected(adj_matrix)

        batch_adj_matrix = build_batch_edge_index(adj_matrix, batch_size, self.node_nums)

        return batch_adj_matrix, adj_matrix

    def _weight_matrix_construct(self, adj, init_edge_index, batch_size):
        mat = to_dense_adj(init_edge_index, edge_attr=adj).squeeze()

        if self.undirected_adj:
            adj = (mat + mat.T) * 0.5
        else:
            adj = mat

        if self.graph_learning_mode == 'attention':
            pass
        else:
            adj = self.correlation_act(adj)

        init_edge_index, adj = dense_to_sparse(adj)

        batch_adj_matrix = build_batch_edge_index(init_edge_index, batch_size, self.node_nums)
        batch_weight_matrix = build_batch_edge_weight(adj, batch_size)

        return batch_adj_matrix, batch_weight_matrix, adj

    def forward(self, inputs, targets, entire_inputs, edge_index):
        batch_size = inputs.shape[0] // self.node_nums
        
        adj = self.graph_learning(entire_inputs, edge_index)

        if (self.graph_learning_mode == 'weight') or (self.graph_learning_mode == 'attention'):
            if self.graph_learning_mode == 'attention':
                edge_index = adj[0]
                adj = adj[1]

            batch_adj_matrix, batch_weight_matrix, adj_matrix = self._weight_matrix_construct(adj, edge_index,
                                                                                              batch_size)

        elif self.graph_learning_mode == 'adj':
            batch_adj_matrix, adj_matrix = self._gumbel_softmax_structure_sampling(adj, edge_index, batch_size)
            batch_weight_matrix = None

            if self.graph_learning_sequence > 1:
                for _ in range(self.graph_learning_sequence-1):
                    adj = self.graph_learning(entire_inputs, adj_matrix)
                    batch_adj_matrix, adj_matrix = self._gumbel_softmax_structure_sampling(adj, adj_matrix, batch_size)

        elif self.graph_learning_mode == 'both':
            batch_adj_matrix, adj_matrix = self._gumbel_softmax_structure_sampling(adj, edge_index, batch_size)

            if self.graph_learning_sequence > 1:
                for _ in range(self.graph_learning_sequence-1):
                    adj = self.graph_learning(entire_inputs, adj_matrix)
                    batch_adj_matrix, adj_matrix = self._gumbel_softmax_structure_sampling(adj, adj_matrix, batch_size)

            _weight = self.weight_learning(entire_inputs, adj_matrix)
            _weight_mat = to_dense_adj(adj_matrix, edge_attr=_weight).squeeze()
            _weight_mat = (_weight_mat + _weight_mat.T) * 0.5
            _weight_mat = self.correlation_softmax(_weight_mat)

            temp = []
            for ii in range(adj_matrix.shape[1]):
                row, col = adj_matrix[0, ii], adj_matrix[1, ii]
                temp.append(_weight_mat[row, col])

            _weight = torch.stack(temp).view(-1, 1)

            batch_weight_matrix = build_batch_edge_weight(_weight, batch_size)

            mat = to_dense_adj(adj_matrix)[0]

            adj_matrix = {'adj_matrix': mat.detach().cpu(), 'weight_matrix': _weight_mat.detach().cpu()}

        else:
            raise ValueError("Invalid graph learning mode")

        outputs = self.graph_forecasting(inputs, targets, batch_adj_matrix, weight_matrix=batch_weight_matrix)

        return adj_matrix, outputs
