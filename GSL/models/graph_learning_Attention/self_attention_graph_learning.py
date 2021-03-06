import torch
import torch.nn as nn
from torch.nn import functional as F
from models.graph_learning_Attention.Attention import GraphLearningMultiHeadAttention


class Attention_Graph_Learning(nn.Module):
    def __init__(self, config):
        super(Attention_Graph_Learning, self).__init__()

        self.num_nodes = config.dataset.nodes_num
        self.nodes_feas = config.dataset.node_features

        self.sampling = config.graph_learning.sampling
        self.total_length = config.dataset.graph_learning_length
        self.kernel_size = config.graph_learning.kernel_size
        self.stride = config.graph_learning.stride
        self.conv_dim = config.graph_learning.conv_dim
        self.hidden_dim = config.graph_learning.hidden_dim

        self.attention_gl = GraphLearningMultiHeadAttention(config.graph_learning.n_head,
                                                            self.hidden_dim,
                                                            self.num_nodes)

        # out_size = 0
        # for i in range(len(self.kernel_size)):
        #     if i == 0:
        #         out_size = int(((self.total_length - self.kernel_size[i]) / self.stride[i]) + 1)
        #     else:
        #         out_size = int(((out_size - self.kernel_size[i]) / self.stride[i]) + 1)

        self.feature_extracotr = nn.ModuleList()
        self.feature_batchnorm = nn.ModuleList()

        for i in range(len(self.conv_dim)):
            if i == 0:
                self.feature_extracotr.append(nn.Conv1d(self.nodes_feas,
                                                        self.conv_dim[i],
                                                        self.kernel_size[i],
                                                        stride=self.stride[i]))
            else:
                self.feature_extracotr.append(nn.Conv1d(self.conv_dim[i - 1],
                                                        self.conv_dim[i],
                                                        self.kernel_size[i],
                                                        stride=self.stride[i]))

        temp_inpt = torch.Tensor(self.num_nodes, self.nodes_feas, self.total_length)
        out_size = []
        for layer in self.feature_extracotr:
            temp_inpt = layer(temp_inpt)
            out_size.append(temp_inpt.shape[-1])

        for i in range(len(self.conv_dim)):
            self.feature_batchnorm.append(nn.LayerNorm(out_size[i]))

        self.fc_concat = nn.Linear(out_size[-1] * self.conv_dim[-1], self.hidden_dim)
        self.hidden_drop = nn.Dropout(0.4)

        self.feature_batchnorm.append(nn.LayerNorm(self.hidden_dim))

        if self.sampling == 'Gumbel_softmax':
            self.gumbel_trick = nn.Linear(1, 2)

    def init_weights(self):
        for m in self.modules():
            if (isinstance(m, nn.Linear)) or (isinstance(m, nn.Conv1d)):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.fill_(0.1)

    def forward(self, x):
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

        outputs, attention_matrix = self.attention_gl(x, x)

        if self.sampling == 'Gumbel_softmax':
            outputs = self.gumbel_trick(outputs.unsqueeze(dim=-1))

        return outputs, attention_matrix
