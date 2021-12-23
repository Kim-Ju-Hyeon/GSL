from models.GTS.gts_graph_learning import GTS_Graph_Learning
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch_geometric_temporal.nn.recurrent import DCRNN
from torch_geometric.utils.random import erdos_renyi_graph

class GTS_Spike_Decoding(nn.Module):
    def __init__(self, config):
        super(GTS_Spike_Decoding, self).__init__()

        self.config = config

        self.num_nodes = config.nodes_num
        self.nodes_feas = config.node_features

        self.kernel_size = config.embedding.kernel_size
        self.stride = config.embedding.stride
        self.conv1_dim = config.embedding.conv_dim[0]
        self.conv2_dim = config.embedding.conv_dim[1]

        self.embedding_dim = config.embedding.embedding_dim

        self.conv1 = torch.nn.Conv1d(self.nodes_feas, self.conv1_dim, self.kernel_size[0],
                                     stride=self.stride[0])
        self.conv2 = torch.nn.Conv1d(self.conv1_dim, self.conv2_dim,
                                     self.kernel_size[1], stride=self.stride[1])

        self.bn1 = torch.nn.BatchNorm1d(self.conv1_dim)
        self.bn2 = torch.nn.BatchNorm1d(self.conv2_dim)

        self.recurrent = DCRNN(self.embedding_dim, self.embedding_dim, self.config.diffusion_k)

    def forward(self, x, edge_index, hidden_state):
        batch_nodes = x.shape[0]
        if len(x.shape) == 2:
            x = x.reshape(batch_nodes, 1, -1)
        x = self.conv1(x)
        x = F.relu(x)
        x = self.bn1(x)
        # print(x.shape)

        x = self.conv2(x)
        x = F.relu(x)
        x = self.bn2(x)
        # print(x.shape)

        x = x.view(batch_nodes, -1)
        # print(x.shape)

        h = self.recurrent(x, edge_index, H=hidden_state)
        h = F.relu(h)

        return h



class GTS_Model(nn.Module):
    def __init__(self, config):
        super(GTS_Model, self).__init__()

        self.config = config
        self.dataset_conf = config.dataset

        self.node_nums = config.nodes_num
        self.embedding_dim = config.embedding.embedding_dim
        self.decode_step = config.decode_step
        self.decode_dim =config.decode_dim

        self.graph_learning = GTS_Graph_Learning(self.config)
        self.graph_forecasting = GTS_Spike_Decoding(self.config)

        self.linear_1 = torch.nn.Linear(self.node_nums*self.embedding_dim, self.node_nums*self.embedding_dim//2)

        self.lin_reg = nn.ModuleList()
        for _ in range(self.decode_dim):
            self.lin_reg.append(torch.nn.Linear(self.node_nums*self.embedding_dim//2, 742))

        self.classification_1 = torch.nn.Linear(self.node_nums*self.embedding_dim//2,
                                              self.node_nums*self.embedding_dim//8)
        self.classification_2 = torch.nn.Linear(self.node_nums * self.embedding_dim // 8, 8)


    def sliding(self):
        valid_sampling_locations = []
        valid_sampling_locations += [
            (self.dataset_conf.window_size + i)
            for i in range(self.dataset_conf.total_time_length - self.dataset_conf.window_size + 1)
            if (i % self.dataset_conf.slide) == 0
        ]

        return valid_sampling_locations

    def forward(self, inputs, edge_index):
        # adj = self.graph_learning(inputs, edge_index)
        #
        # edge_probability = F.gumbel_softmax(adj, tau=0.3, hard=True)
        # edge_probability = torch.transpose(edge_probability, 0, 1)
        #
        # edge_ = []
        # for ii, rel in enumerate(edge_probability[0]):
        #     if bool(rel):
        #         edge_.append(edge_index[:, ii])
        #
        # adj_matrix = torch.stack(edge_, dim=-1)

        adj_matrix =erdos_renyi_graph(self.node_nums, 0.1)

        loc = self.sliding()

        hidden_state=None
        for start_idx in loc:
            spike_inputs_per_window = inputs[:, start_idx - self.dataset_conf.window_size:start_idx]
            hidden_state = self.graph_forecasting(spike_inputs_per_window, adj_matrix, hidden_state)

        hidden_state = hidden_state.view(-1)

        out = self.linear_1(hidden_state)

        angle_out = self.classification_1(out)
        angle = self.classification_2(angle_out)

        outputs = []
        for ii in range(self.decode_dim):
            output = self.lin_reg[ii](out)
            outputs.append(output)

        outputs = torch.stack(outputs, dim=0)

        return adj_matrix, outputs, angle
