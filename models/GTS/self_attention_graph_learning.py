import torch
import torch.nn as nn
from torch.nn import functional as F
from models.Attention import GraphLearningMultiHeadAttention


class Attention_Graph_Learning(nn.Module):
    def __init__(self, config):
        super(Attention_Graph_Learning, self).__init__()
        self.num_nodes = config.nodes_num
        self.nodes_feas = config.node_features

        self.mode = config.graph_learning.mode

        self.total_length = config.dataset.graph_learning_length
        self.kernel_size = config.graph_learning.kernel_size
        self.stride = config.graph_learning.stride
        self.conv1_dim = config.graph_learning.conv1_dim
        self.conv2_dim = config.graph_learning.conv2_dim
        self.conv3_dim = config.graph_learning.conv3_dim
        self.hidden_dim = config.hidden_dim

        self.attention_gl = GraphLearningMultiHeadAttention(config.graph_learning.n_head,
                                                            self.hidden_dim,
                                                            self.num_nodes)

        out_size = 0
        for i in range(len(self.kernel_size)):
            if i == 0:
                out_size = int(((self.total_length - self.kernel_size[i]) / self.stride[i]) + 1)
            else:
                out_size = int(((out_size - self.kernel_size[i]) / self.stride[i]) + 1)

        self.conv1 = torch.nn.Conv1d(self.nodes_feas, self.conv1_dim, self.kernel_size[0], stride=self.stride[0])
        self.conv2 = torch.nn.Conv1d(self.conv1_dim, self.conv2_dim, self.kernel_size[1], stride=self.stride[1])
        self.conv3 = torch.nn.Conv1d(self.conv2_dim, self.conv3_dim, self.kernel_size[2], stride=self.stride[2])

        self.fc_conv = torch.nn.Conv1d(self.conv3_dim, 1, 1, stride=1)

        self.fc = nn.Linear(out_size, self.hidden_dim)

        self.hidden_drop = nn.Dropout(0.4)

        self.bn1 = torch.nn.BatchNorm1d(self.conv1_dim)
        self.bn2 = torch.nn.BatchNorm1d(self.conv2_dim)
        self.bn3 = torch.nn.BatchNorm1d(self.conv3_dim)

    def init_weights(self):
        for m in self.modules():
            if (isinstance(m, nn.Linear)) or (isinstance(m, nn.Conv1d)):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.fill_(0.1)

    def forward(self, x, edge_index):
        if len(x.shape) == 2:
            x = x.reshape(self.num_nodes, 1, -1)

        x = self.conv1(x)
        x = F.relu(x)
        x = self.bn1(x)

        x = self.conv2(x)
        x = F.relu(x)
        x = self.bn2(x)

        x = self.conv3(x)
        x = F.relu(x)
        x = self.bn3(x)

        x = self.fc_conv(x)
        x = F.relu(x)

        x = x.squeeze()
        x = self.fc(x)
        x = F.relu(x)

        outputs = self.attention_gl(x, x)

        return outputs
