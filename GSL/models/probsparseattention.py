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

    def _update_context(self, init_attn, scores, index):
        B, H, _, _ = init_attn.shape
        attn = torch.softmax(scores, dim=-1)  # nn.Softmax(dim=-1)(scores)

        init_attn[torch.arange(B)[:, None, None], torch.arange(H)[None, :, None], index, :] = attn

        return init_attn

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
        scale = 1. / sqrt(D)
        scores_top = scores_top * scale

        # update the context with selected top_k queries
        attn = torch.softmax(scores_top, dim=-1)

        attention_matirx = (torch.ones([B, H, L, L]) / L).type_as(attn).to(scores_top.device)
        attention_matirx[torch.arange(B)[:, None, None], torch.arange(H)[None, :, None], index, :] = attn
        return attention_matirx


class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads,
                 d_keys=None, d_values=None, mix=False):
        super(AttentionLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads
        self.mix = mix

    def forward(self, queries, keys, values, attn_mask):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        out, attn = self.inner_attention(
            queries,
            keys,
            values,
            attn_mask
        )
        if self.mix:
            out = out.transpose(2, 1).contiguous()
        out = out.view(B, L, -1)

        return self.out_projection(out), attn


class GraphLearningProbSparseAttention(nn.Module):
    def __init__(self, n_head, d_model, num_nodes, dropout_rate=0.5, factor=2):
        super(GraphLearningProbSparseAttention, self).__init__()

        self.n_head = n_head
        self.d_model = d_model
        self.num_nodes = num_nodes

        self.d_k = self.d_q = d_model // n_head
        self.dropout = nn.Dropout(p=dropout_rate)

        self.query_projection = nn.Linear(self.d_model, self.d_k * self.n_head)
        self.key_projection = nn.Linear(self.d_model, self.d_k * self.n_head)

        self.attention = ProbAttention(factor)

        self.init_weights()

    def init_weights(self):
        for name, p in self.named_parameters():
            if 'bias' not in name:
                torch.nn.init.xavier_uniform_(p)
            else:
                torch.nn.init.zeros_(p)

    def forward(self, q, k):
        B, nodes_num, hidden = q.shape

        queries = self.query_projection(q).view(B, nodes_num, self.n_head, -1)
        keys = self.key_projection(k).view(B, nodes_num, self.n_head, -1)

        attn = self.attention(queries, keys)

        return attn
