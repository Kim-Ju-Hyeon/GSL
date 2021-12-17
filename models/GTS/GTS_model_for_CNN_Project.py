from models.GTS.gts_graph_learning import GTS_Graph_Learning
from models.GTS.gts_forecasting_module import GTS_Spike_Decoding
from utils.utils import build_dynamic_batch_edge_index

import torch
import torch.nn as nn
from torch.nn import functional as F


class GTS_Model(nn.Module):
    def __init__(self, config):
        super(GTS_Model, self).__init__()

        self.config = config

        self.node_nums = config.nodes_num

        self.graph_learning = GTS_Graph_Learning(self.config)
        self.graph_forecasting = GTS_Spike_Decoding(self.config)

        self.loss = config.train.loss_function

        if self.loss == 'CrossEntropy':
            self.loss_func = nn.CrossEntropyLoss()

        elif self.loss == 'MSELoss':
            self.loss_func = nn.MSELoss()

        else:
            raise ValueError("Non-supported loss function!")

    def forward(self, inputs, targets, entire_inputs, edge_index):
        batch_size = inputs.shape[0] // self.node_nums

        edge_list = []
        for batch in range(batch_size):
            adj = self.graph_learning(entire_inputs[batch,:,:], edge_index)

            edge_probability = F.gumbel_softmax(adj, tau=0.3, hard=True)
            edge_probability = torch.transpose(edge_probability, 0, 1)

            edge_ = []
            for ii, rel in enumerate(edge_probability[0]):
                if bool(rel):
                    edge_.append(edge_index[:, ii])

            adj_matrix = torch.stack(edge_, dim=-1)

            edge_list.append(adj_matrix)

        batch_adj_matrix = build_dynamic_batch_edge_index(edge_list)

        outputs = self.graph_forecasting(inputs, batch_adj_matrix)

        loss = self.loss_func(outputs, targets)

        return edge_list, outputs, loss
