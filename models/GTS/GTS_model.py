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

    def forward(self, inputs, targets, entire_inputs, edge_index):
        batch_size = inputs.shape[0] // self.node_nums
        
        print(edge_index.shape)
        adj = self.graph_learning(entire_inputs, edge_index)
        print(adj.shape)

        if self.graph_learning_mode == 'weight':
            adj_matrix = adj

            if self.undirected_adj:
                mat = to_dense_adj(edge_index, edge_attr=adj).squeeze()
                mat = self.correlation_softmax(mat)
                adj = (mat + mat.T) * 0.5
                adj_matrix = adj
                edge_self_loop = add_self_loops(edge_index)
                edge_index = sort_edge_index(edge_self_loop[0])
                adj = adj.view(-1, 1)
                edge_index, adj = remove_self_loops(edge_index, adj)

            batch_adj_matrix = build_batch_edge_index(edge_index, batch_size, self.node_nums)
            batch_weight_matrix = build_batch_edge_weight(adj, batch_size)

            outputs = self.graph_forecasting(inputs, targets, batch_adj_matrix, weight_matrix=batch_weight_matrix)

        elif self.graph_learning_mode == 'adj':
            batch_adj_matrix, adj_matrix = self._gumbel_softmax_structure_sampling(adj, edge_index, batch_size)
            if self.graph_learning_sequence > 1:
                for _ in range(self.graph_learning_sequence-1):
                    adj = self.graph_learning(entire_inputs, adj_matrix)
                    print(adj.shape)
                batch_adj_matrix, adj_matrix = self._gumbel_softmax_structure_sampling(adj, edge_index, batch_size)

            outputs = self.graph_forecasting(inputs, targets, batch_adj_matrix)
        else:
            raise ValueError("Invalid graph learning mode")

        return adj_matrix, outputs
