from __future__ import division

import numbers
from typing import Optional

import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F


class MTGNN_Graph_Learning(nn.Module):
    def __init__(self, config):
        super(MTGNN_Graph_Learning, self).__init__()

        self.num_nodes = config.dataset.nodes_num
        dim = config.graph_learning.hidden_dim

        self.device = config.device

        self.sampling = config.graph_learning.sampling

        self._embedding1 = nn.Embedding(self.num_nodes, dim)
        self._embedding2 = nn.Embedding(self.num_nodes, dim)
        self._linear1 = nn.Linear(dim, dim)
        self._linear2 = nn.Linear(dim, dim)

        if self.sampling == 'Gumbel_softmax':
            self.gumbel_trick = nn.Linear(1, 2)

        self._alpha = config.graph_learning.alpha

        self._reset_parameters()


    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            else:
                nn.init.uniform_(p)

    def forward(self) -> torch.Tensor:
        nodevec1 = self._embedding1(torch.arange(self.num_nodes).to(self.device))
        nodevec2 = self._embedding2(torch.arange(self.num_nodes).to(self.device))

        nodevec1 = torch.tanh(self._alpha * self._linear1(nodevec1))
        nodevec2 = torch.tanh(self._alpha * self._linear2(nodevec2))

        outputs = torch.mm(nodevec1, nodevec2.transpose(1, 0)) - torch.mm(
            nodevec2, nodevec1.transpose(1, 0)
        )

        if self.sampling == 'Gumbel_softmax':
            outputs = self.gumbel_trick(outputs.unsqueeze(dim=-1))

        outputs = F.relu(torch.tanh(self._alpha * outputs))

        return outputs
