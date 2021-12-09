import torch
import torch.nn as nn
from torch.nn import functional as F
from torch_geometric.nn import MessagePassing
import math


class GTS_Graph_Learning(MessagePassing):
    def __init__(self, config):
        super(GTS_Graph_Learning, self).__init__(aggr=None)

        self.hidden_dim = config.hidden_dim
        self.num_nodes = config.nodes_num
        self.nodes_feas = config.node_features

        self.kernel_size = config.kernel_size
        self.stride = config.stride
        self.conv1_dim = config.conv1_dim
        self.conv2_dim = config.conv2_dim
        self.fc_dim = config.fc_dim

        self.conv1 = torch.nn.Conv1d(self.nodes_feas, self.conv1_dim, self.kernel_size, stride=self.stride)
        self.conv2 = torch.nn.Conv1d(self.conv1_dim, self.conv2_dim, self.kernel_size, stride=self.stride)

        self.hidden_drop = nn.Dropout(0.2)

        self.fc = torch.nn.Linear(self.fc_dim, self.hidden_dim)

        self.bn1 = torch.nn.BatchNorm1d(self.conv1_dim)
        self.bn2 = torch.nn.BatchNorm1d(self.conv2_dim)
        self.bn3 = torch.nn.BatchNorm1d(self.hidden_dim)

        self.fc_out = nn.Linear(self.hidden_dim * 2, self.hidden_dim)
        self.fc_cat = nn.Linear(self.hidden_dim, 2)

        # self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if (isinstance(m, nn.Linear)) or (isinstance(m, nn.Conv1d)):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.fill_(0.1)

    def forward(self, x, edge_index):
        x = x.transpose(1, 0).reshape(self.num_nodes, 1, -1)  # Input Data에 따라 수정해야됨 --- Feature dim 고려

        x = self.conv1(x)
        x = F.relu(x)
        x = self.bn1(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.bn2(x)

        x = x.view(self.num_nodes, -1)

        x = self.fc(x)
        x = F.relu(x)
        x = self.bn3(x)

        x = self.propagate(edge_index, x=x)
        return x

    def message(self, x_i, x_j):
        x = torch.cat([x_i, x_j], dim=-1)
        return x

    def update(self, aggr_out):
        x = self.fc_cat(aggr_out)
        return x

    def aggregate(self, x):
        x = torch.relu(self.fc_out(x))
        return x
