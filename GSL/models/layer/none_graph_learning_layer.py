import torch
import torch.nn as nn
from torch_geometric.utils import erdos_renyi_graph, dense_to_sparse
import numpy as np
from utils.utils import build_fully_connected_edge_idx


class None_Graph_Learning(nn.Module):
    def __init__(self, config):
        super(None_Graph_Learning, self).__init__()
        self.config = config

        self.graph_mode = config.graph_learning.graph_mode
        self.nodes_num = config.dataset.nodes_num
        self.edge_prob = config.graph_learning.edge_prob

    def forward(self):
        if self.graph_mode == 'random_graph':
            _edge_index = erdos_renyi_graph(num_nodes=self.nodes_num, edge_prob=self.edge_prob)
            _edge_attr = None

        elif self.graph_mode == 'complete_graph':
            _edge_index = build_fully_connected_edge_idx(num_nodes=self.nodes_num)
            _edge_attr = None

        elif self.graph_mode == 'no_graph':
            _edge_index = torch.arange(0, self.nodes_num, dtype=torch.long)
            _edge_index = _edge_index.unsqueeze(0).repeat(2, 1)
            _edge_attr = None

        elif self.graph_mode == 'ground_truth':
            if self.config.dataset.name == 'PEMS_BAY':
                adj_matrix = np.load('./data/PEMS-BAY/pems_adj_mat.npy')
            elif self.config.dataset.name == 'METR-LA':
                adj_matrix = np.load('./data/METR-LA/adj_mat.npy')
            else:
                raise ValueError("No Ground truth Graph")

            _edge_index, _edge_attr = dense_to_sparse(torch.Tensor(adj_matrix))
            _edge_attr = _edge_attr.to(self.config.device)
        else:
            raise ValueError("Invalid graph mode")

        _edge_index = _edge_index.to(self.config.device)

        return _edge_index, _edge_attr
