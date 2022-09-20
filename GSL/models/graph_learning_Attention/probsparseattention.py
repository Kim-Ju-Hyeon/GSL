import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from math import sqrt


class ProbAttention(nn.Module):
    def __init__(self, factor):
        super(ProbAttention, self).__init__()

        self.factor = factor

    def _prob_QK(self, Q, K, sample_k, n_top):  # n_top: c*ln(L_q)
        B, H, L, E = K.shape
        _, _, L, _ = Q.shape

        # calculate the sampled Q_K
        K_expand = K.unsqueeze(-3).expand(B, H, L, L, E)
        index_sample = torch.randint(L, (L, sample_k))  # real U = U_part(factor*ln(L_k))*L_q
        K_sample = K_expand[:, :, torch.arange(L).unsqueeze(1), index_sample, :]
        Q_K_sample = torch.matmul(Q.unsqueeze(-2), K_sample.transpose(-2, -1)).squeeze(-2)

        # find the Top_k query with sparisty measurement
        M = Q_K_sample.max(-1)[0] - torch.div(Q_K_sample.sum(-1), L)
        M_top = M.topk(n_top, sorted=False)[1]

        # use the reduced Q to calculate Q_K
        Q_reduce = Q[torch.arange(B)[:, None, None],
                     torch.arange(H)[None, :, None],
                     M_top, :]  # factor*ln(L_q)

        Q_K = torch.matmul(Q_reduce, K.transpose(-2, -1))  # factor*ln(L_q)*L_k

        return Q_K, M_top

    def forward(self, queries, keys):
        B, L, H, D = queries.shape

        queries = queries.transpose(2, 1)
        keys = keys.transpose(2, 1)

        U_part = 2 * self.factor * np.ceil(np.log(L)).astype('int').item()  # c*ln(L_k)
        u = self.factor * np.ceil(np.log(L)).astype('int').item()  # c*ln(L_q)

        U_part = U_part if U_part < L else L
        u = u if u < L else L

        scores_top, index = self._prob_QK(queries, keys, sample_k=U_part, n_top=u)

        # add scale factor
        scale = 1. / torch.sqrt(torch.tensor(keys.shape[-1]).to(torch.float32))
        scores_top = scores_top * scale

        # update the context with selected top_k queries
        attn = torch.softmax(scores_top, dim=-1)

        attention_matirx = (torch.zeros([B, H, L, L])).type_as(attn).to(scores_top.device)
        attention_matirx[torch.arange(B)[:, None, None], torch.arange(H)[None, :, None], index, :] = attn

        return attention_matirx


class GraphLearningProbSparseAttention(nn.Module):
    def __init__(self, config):
        super(GraphLearningProbSparseAttention, self).__init__()
        self.num_nodes = config.dataset.nodes_num
        self.batch_size = config.train.batch_size

        self.n_head = config.graph_learning.n_head
        self.sequence_length = config.forecasting_module.backcast_length
        self.dropout = nn.Dropout(p=config.graph_learning.dropout_rate)

        self.kernel_size = config.graph_learning.kernel_size
        self.stride = config.graph_learning.stride
        self.conv_dim = config.graph_learning.conv_dim
        self.hidden_dim = config.graph_learning.hidden_dim

        self.feature_extracotr = nn.ModuleList()
        self.feature_batchnorm = nn.ModuleList()

        for i in range(len(self.conv_dim)):
            if i == 0:
                self.feature_extracotr.append(nn.Conv1d(1,
                                                        self.conv_dim[i],
                                                        self.kernel_size[i],
                                                        stride=self.stride[i]))
            else:
                self.feature_extracotr.append(nn.Conv1d(self.conv_dim[i-1],
                                                        self.conv_dim[i],
                                                        self.kernel_size[i],
                                                        stride=self.stride[i]))

        temp_inpt = torch.Tensor(self.num_nodes*self.batch_size, 1, self.sequence_length)
        out_size = []
        for layer in self.feature_extracotr:
            temp_inpt = layer(temp_inpt)
            out_size.append(temp_inpt.shape[-1])

        for i in range(len(self.conv_dim)):
            self.feature_batchnorm.append(nn.LayerNorm(out_size[i]))

        self.fc_concat = nn.Linear(out_size[-1] * self.conv_dim[-1], self.hidden_dim)
        self.feature_batchnorm.append(nn.LayerNorm(self.hidden_dim))

        self.d_k = self.d_q = self.hidden_dim // self.n_head
        self.query_projection = nn.Linear(self.hidden_dim, self.d_k * self.n_head)
        self.key_projection = nn.Linear(self.hidden_dim, self.d_k * self.n_head)

        self.attention = ProbAttention(config.graph_learning.factor)

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if (isinstance(m, nn.Linear)) or (isinstance(m, nn.Conv1d)):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.fill_(0.1)

    def forward(self, x):
        B, nodes_num, hidden = x.shape

        x = x.view(B*nodes_num, 1, -1)
        for i, conv in enumerate(self.feature_extracotr):
            x = conv(x)
            x = F.relu(x)
            x = self.feature_batchnorm[i](x)

        x = x.view(self.num_nodes*self.batch_size, -1)

        x = self.fc_concat(x)
        x = F.relu(x)
        x = self.feature_batchnorm[-1](x)

        queries = self.query_projection(x).view(B, nodes_num, self.n_head, -1)
        keys = self.key_projection(x).view(B, nodes_num, self.n_head, -1)

        attn = self.attention(queries, keys)
        attn = attn.masked_fill(attn < 1 / nodes_num, 0)
        attn = attn.mean(dim=1)

        # attn = attn.masked_fill(attn < 1/nodes_num, 0)

        return attn
