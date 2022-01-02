from models.GTS.gts_graph_learning import GTS_Graph_Learning
from models.GTS.gts_forecasting_module import GTS_Forecasting_Module
from utils.utils import build_batch_edge_index

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch_geometric.utils import to_undirected


class GTS_Model(nn.Module):
    def __init__(self, config):
        super(GTS_Model, self).__init__()

        self.config = config

        self.node_nums = config.nodes_num

        self.undirected_adj = config.graph_learning.to_symmetric

        self.graph_learning = GTS_Graph_Learning(self.config)
        self.graph_forecasting = GTS_Forecasting_Module(self.config)

        self.loss = config.train.loss_function

        if self.loss == 'CrossEntropy':
            self.loss_func = nn.CrossEntropyLoss()

        elif self.loss == 'MSELoss':
            self.loss_func = nn.MSELoss()

        # elif self.loss == 'poissinLoss':
        #     self.loss_func = F.poisson_nll_loss()

        else:
            raise ValueError("Non-supported loss function!")

    def forward(self, inputs, targets, entire_inputs, edge_index):
        batch_size = inputs.shape[0] // self.node_nums

        adj = self.graph_learning(entire_inputs, edge_index)

        edge_probability = F.gumbel_softmax(adj, tau=0.3, hard=True)
        connect = torch.where(edge_probability[:, 0])

        adj_matrix = torch.stack([edge_index[0, :][connect], edge_index[1, :][connect]])
        if self.undirected_adj:
            adj_matrix = to_undirected(adj_matrix)
        batch_adj_matrix = build_batch_edge_index(adj_matrix, batch_size)

        outputs = self.graph_forecasting(inputs, targets, batch_adj_matrix)

        loss = self.loss_func(outputs, targets)

        return adj_matrix, outputs, loss
