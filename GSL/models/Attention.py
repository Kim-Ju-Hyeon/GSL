from torch import nn
import torch
from torch.nn import functional as F


class GraphLearningScaledDotProductAttention(nn.Module):
    def __init__(self, dropout=0.4):
        super(GraphLearningScaledDotProductAttention, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, q, k, mask=None):
        attn = torch.matmul(q, k.transpose(0, 1))

        dimention = torch.sqrt(torch.tensor(k.shape[-1]).to(torch.float32))
        attn = attn / dimention

        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)

        attn = self.softmax(attn)
        attn = self.dropout(attn)

        return attn


class GraphLearningMultiHeadAttention(nn.Module):
    def __init__(self, n_head, d_model, num_nodes, dropout_rate=0.5):
        super(GraphLearningMultiHeadAttention, self).__init__()

        self.n_head = n_head
        self.d_model = d_model
        self.num_nodes = num_nodes

        self.d_k = self.d_q = d_model // n_head
        self.dropout = nn.Dropout(p=dropout_rate)

        self.q_layers = nn.ModuleList([nn.Linear(self.d_model, self.d_q, bias=False)
                                       for _ in range(self.n_head)])
        self.k_layers = nn.ModuleList([nn.Linear(self.d_model, self.d_k, bias=False)
                                       for _ in range(self.n_head)])

        self.attention = GraphLearningScaledDotProductAttention()

        self.output = nn.Linear(self.num_nodes * n_head, self.num_nodes)

        self.init_weights()

    def init_weights(self):
        for name, p in self.named_parameters():
            if 'bias' not in name:
                torch.nn.init.xavier_uniform_(p)
            else:
                torch.nn.init.zeros_(p)

    def forward(self, q, k, mask=None):
        nodes_num, _ = q.shape
        _attns = []
        for i in range(self.n_head):
            qs = F.relu(self.q_layers[i](q))
            ks = F.relu(self.k_layers[i](k))

            attn = self.attention(qs, ks, mask)

            attn_dropout = self.dropout(attn)
            _attns.append(attn_dropout)

        attn = torch.stack(_attns)
        attn = attn.reshape(nodes_num, -1)
        outputs = self.output(attn)

        return outputs, _attns
