from models.GTS.gts_graph_learning import GTS_Graph_Learning
from models.GTS.gts_forecasting_module import GTS_Forecasting_Module
from utils.utils import build_batch_edge_index, build_batch_edge_weight

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch_geometric.utils import to_undirected, to_dense_adj, add_self_loops, sort_edge_index, remove_self_loops


class GTS_Model(nn.Module):
    def __init__(self, config):
        super(GTS_Model, self).__init__()

        self.config = config

        self.node_nums = config.nodes_num
        self.tau = config.tau

        self.undirected_adj = config.graph_learning.to_symmetric

        self.graph_learning_mode = config.graph_learning.mode
        self.graph_learning_sequence = config.graph_learning.sequence

        self.graph_learning = GTS_Graph_Learning(self.config)
        self.graph_forecasting = GTS_Forecasting_Module(self.config)

        self.correlation_softmax = nn.Softmax(dim=1)

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

        adj = self.correlation_softmax(adj)
        edge_self_loop = add_self_loops(init_edge_index)
        init_edge_index = sort_edge_index(edge_self_loop[0])
        adj = adj.view(-1, 1)
        init_edge_index, adj = remove_self_loops(init_edge_index, adj)

        batch_adj_matrix = build_batch_edge_index(init_edge_index, batch_size, self.node_nums)
        batch_weight_matrix = build_batch_edge_weight(adj, batch_size)

        return batch_adj_matrix, batch_weight_matrix, adj

    def forward(self, inputs, targets, entire_inputs, edge_index):
        batch_size = inputs.shape[0] // self.node_nums
        
        adj = self.graph_learning(entire_inputs, edge_index)

        if self.graph_learning_mode == 'weight':
            batch_adj_matrix, batch_weight_matrix, adj_matrix = self._weight_matrix_construct(adj, edge_index, batch_size)

        elif self.graph_learning_mode == 'adj':
            batch_adj_matrix, adj_matrix = self._gumbel_softmax_structure_sampling(adj, edge_index, batch_size)
            batch_weight_matrix = None

            if self.graph_learning_sequence > 1:
                for _ in range(self.graph_learning_sequence-1):
                    adj = self.graph_learning(entire_inputs, adj_matrix)
                    batch_adj_matrix, adj_matrix = self._gumbel_softmax_structure_sampling(adj, adj_matrix, batch_size)
                    print(adj_matrix.shape)
        else:
            raise ValueError("Invalid graph learning mode")

        outputs = self.graph_forecasting(inputs, targets, batch_adj_matrix, weight_matrix=batch_weight_matrix)

        return adj_matrix, outputs
