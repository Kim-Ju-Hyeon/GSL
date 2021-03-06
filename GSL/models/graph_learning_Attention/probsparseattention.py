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

        U_part = self.factor * np.ceil(np.log(L)).astype('int').item()  # c*ln(L_k)
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

        self.n_head = config.graph_learning.n_head
        self.sequence_length = config.forecasting_module.backcast_length
        self.dropout = nn.Dropout(p=config.graph_learning.dropout_rate)
        self._mlp_layers = config.graph_learning.pre_mlp_layer

        self.mlp_layers = nn.ModuleList()
        for i in range(len(self._mlp_layers)):
            if i == 0:
                self.mlp_layers.append(nn.Linear(self.sequence_length, self._mlp_layers[i]))
            else:
                self.mlp_layers.append(nn.Linear(self._mlp_layers[i-1], self._mlp_layers[i]))

        if len(self._mlp_layers) == 0:
            self.d_model = self.sequence_length
        else:
            self.d_model = self._mlp_layers[-1]

        self.d_k = self.d_q = self.d_model // self.n_head
        self.query_projection = nn.Linear(self.d_model, self.d_k * self.n_head)
        self.key_projection = nn.Linear(self.d_model, self.d_k * self.n_head)

        self.attention = ProbAttention(config.graph_learning.factor)

        self.init_weights()

    def init_weights(self):
        for name, p in self.named_parameters():
            if 'bias' not in name:
                torch.nn.init.xavier_uniform_(p)
            else:
                torch.nn.init.zeros_(p)

    def forward(self, x):
        B, nodes_num, hidden = x.shape

        for layer in self.mlp_layers:
            x = layer(x)
            x = F.relu(x)

        queries = self.query_projection(x).view(B, nodes_num, self.n_head, -1)
        keys = self.key_projection(x).view(B, nodes_num, self.n_head, -1)

        attn = self.attention(queries, keys)

        return attn
