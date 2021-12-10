from models.GTS.gts_graph_learning import GTS_Graph_Learning
from models.GTS.gts_forecasting_module import GTS_Forecasting_Module
from utils.utils import build_edge_idx

import torch
import torch.nn as nn
from torch.nn import functional as F


class GTS_Model(nn.Module):
    def __init__(self, config):
        super(GTS_Model, self).__init__()

        self.config = config

        self.graph_learning = GTS_Graph_Learning(self.config)
        self.graph_forecasting = GTS_Forecasting_Module(self.config)

        self.loss = config.loss_function

        self.fully_connected_edge_index = build_edge_idx(self.config.num_nodes)

        if self.loss == 'CrossEntropy':
            self.loss_func = nn.CrossEntropyLoss()

        elif self.loss == 'MSELoss':
            self.loss_func = nn.MSELoss()

        else:
            raise ValueError("Non-supported loss function!")

    def forward(self, inputs, entire_inputs, targets):
        adj = self.graph_learning(entire_inputs, self.fully_connected_edge_index)

        edge_probability = F.gumbel_softmax(adj, tau=0.3, hard=True)
        edge_probability = torch.transpose(edge_probability, 0, 1)

        edge_ = []
        for ii, rel in enumerate(edge_probability[0]):
            if bool(rel):
                edge_.append(self.fully_connected_edge_index[:, ii])

        adj_matrix = torch.stack(edge_, dim=-1)

        outputs = self.graph_forecasting(inputs, targets, adj_matrix)

        loss = self.loss_func(outputs, targets)

        return adj_matrix, outputs, loss
