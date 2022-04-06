import torch
import torch.nn as nn
from torch.nn import functional as F
from models.message_passing.message_layer import MessageLayer
from torch_geometric.utils import to_dense_adj, dense_to_sparse


class GTS_Graph_Learning(MessageLayer):
    def __init__(self, config):
        super(GTS_Graph_Learning, self).__init__()
        self.num_nodes = config.nodes_num
        self.nodes_feas = config.node_features

        self.total_length = config.dataset.graph_learning_length
        self.kernel_size = config.graph_learning.kernel_size
        self.stride = config.graph_learning.stride
        self.conv_dim = config.graph_learning.conv_dim
        self.hidden_dim = config.graph_learning.hidden_dim

        if config.graph_learning.sampling == 'Gumbel_softmax':
            out_dim = 2
        else:
            out_dim = 1

        out_size = 0
        for i in range(len(self.kernel_size)):
            if i == 0:
                out_size = int(((self.total_length - self.kernel_size[i]) / self.stride[i]) + 1)
            else:
                out_size = int(((out_size - self.kernel_size[i]) / self.stride[i]) + 1)

        self.feature_extracotr = nn.ModuleList()
        self.feature_batchnorm = nn.ModuleList()

        for i in range(len(self.conv_dim)):
            if i == 0:
                self.feature_extracotr.append(nn.Conv1d(self.nodes_feas,
                                                        self.conv_dim[i],
                                                        self.kernel_size[i],
                                                        stride=self.stride[i]))
            else:
                self.feature_extracotr.append(nn.Conv1d(self.conv_dim[i-1],
                                                        self.conv_dim[i],
                                                        self.kernel_size[i],
                                                        stride=self.stride[i]))

            self.feature_batchnorm.append(nn.BatchNorm1d(self.conv_dim[i]))
        self.feature_batchnorm.append(nn.BatchNorm1d(self.hidden_dim))

        self.fc_concat = nn.Linear(out_size*self.conv_dim[-1], self.hidden_dim)

        self.fc_cat = nn.Linear(self.hidden_dim*2, self.hidden_dim)
        self.fc_out = nn.Linear(self.hidden_dim, out_dim)

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if (isinstance(m, nn.Linear)) or (isinstance(m, nn.Conv1d)):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.fill_(0.1)

    def forward(self, x, edge_index):
        if len(x.shape) == 2:
            x = x.reshape(self.num_nodes, self.nodes_feas, -1)

        for i, conv in enumerate(self.feature_extracotr):
            x = conv(x)
            x = F.relu(x)
            x = self.feature_batchnorm[i](x)

        x = x.view(self.num_nodes, -1)

        x = self.fc_concat(x)
        x = F.relu(x)
        x = self.feature_batchnorm[-1](x)

        _, x = self.propagate(edge_index, x=x)

        return x

    def message(self, x_i, x_j):
        x = torch.cat([x_i, x_j], dim=-1)
        x = F.relu(self.fc_cat(x))
        x = F.relu(self.fc_out(x))
        return x
