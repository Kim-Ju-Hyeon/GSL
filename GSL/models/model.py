from models.GTS.gts_graph_learning import GTS_Graph_Learning
from models.GTS.gts_forecasting_module import GTS_Forecasting_Module, GTS_Traffic_Forecasting_Module
from models.GTS.self_attention_graph_learning import Attention_Graph_Learning
from models.MTGNN.mtgnn_graph_learning import MTGNN_Graph_Learning
from utils.utils import build_batch_edge_index, build_batch_edge_weight

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
        self.graph_learning_sequence = config.graph_learning.sequence

        if self.dataset_conf.name == 'spike_lambda_bin100':
            self.graph_forecasting = GTS_Forecasting_Module(self.config)
        elif (self.dataset_conf.name == 'METR-LA') or (self.dataset_conf.name == 'PEMS-BAY'):
            self.graph_forecasting = GTS_Traffic_Forecasting_Module(self.config)
        else:
            raise ValueError("Non-supported dataset!")

        if self.graph_learning_mode == 'GTS':
            if self.sampling_mode == 'Gumbel_softmax':
                self.graph_learning = GTS_Graph_Learning(self.config, 2)
            else:
                self.graph_learning = GTS_Graph_Learning(self.config, 1)

        elif self.graph_learning_mode == 'attention':
            self.graph_learning = Attention_Graph_Learning(self.config)

        elif self.graph_learning_mode == 'MTGNN':
            self.graph_learning = MTGNN_Graph_Learning(num_nodes=self.node_nums, k=self.graph_learning_parameter.top_k, dim=self.graph_learning_parameter.hidden_dim)
        else:
            raise ValueError("Invalid graph learning mode")