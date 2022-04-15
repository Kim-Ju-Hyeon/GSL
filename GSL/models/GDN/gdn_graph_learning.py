from __future__ import division

import torch
import torch.nn as nn


class GDN_Graph_Learning(nn.Module):
    def __init__(self, config):
        super(GDN_Graph_Learning, self).__init__()

        self.num_nodes = config.nodes_num
        self.dim = config.graph_learning.hidden_dim

        self.sampling = config.graph_learning.sampling

        self._embedding1 = nn.Embedding(self.num_nodes, self.dim)

        if self.sampling == 'Gumbel_softmax':
            self.gumbel_trick = nn.Linear(1, 2)

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            else:
                nn.init.uniform_(p)

    def forward(self) -> torch.Tensor:
        nodevec = self._embedding1(torch.arange(self.num_nodes))
        relation_matrix = torch.matmul(nodevec, nodevec.transpose(0, 1))
        normed_mat = torch.matmul(nodevec.norm(dim=-1).view(-1, 1), nodevec.norm(dim=-1).view(1, -1))

        outputs = relation_matrix / normed_mat

        if self.sampling == 'Gumbel_softmax':
            outputs = self.gumbel_trick(outputs.unsqueeze(dim=-1))

        return outputs