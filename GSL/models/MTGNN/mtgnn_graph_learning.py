from __future__ import division

import numbers
from typing import Optional

import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F


class MTGNN_Graph_Learning(nn.Module):
    def __init__(
            self, num_nodes: int, k: int, dim: int, alpha: float, xd: Optional[int] = None
    ):
        super(MTGNN_Graph_Learning, self).__init__()
        if xd is not None:
            self._static_feature_dim = xd
            self._linear1 = nn.Linear(xd, dim)
            self._linear2 = nn.Linear(xd, dim)
        else:
            self._embedding1 = nn.Embedding(num_nodes, dim)
            self._embedding2 = nn.Embedding(num_nodes, dim)
            self._linear1 = nn.Linear(dim, dim)
            self._linear2 = nn.Linear(dim, dim)

        self._k = k
        self._alpha = alpha

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            else:
                nn.init.uniform_(p)

    def forward(
            self, idx: torch.LongTensor, FE: Optional[torch.FloatTensor] = None
    ) -> torch.FloatTensor:

        if FE is None:
            nodevec1 = self._embedding1(idx)
            nodevec2 = self._embedding2(idx)
        else:
            assert FE.shape[1] == self._static_feature_dim
            nodevec1 = FE[idx, :]
            nodevec2 = nodevec1

        nodevec1 = torch.tanh(self._alpha * self._linear1(nodevec1))
        nodevec2 = torch.tanh(self._alpha * self._linear2(nodevec2))

        a = torch.mm(nodevec1, nodevec2.transpose(1, 0)) - torch.mm(
            nodevec2, nodevec1.transpose(1, 0)
        )
        A = F.relu(torch.tanh(self._alpha * a))
        mask = torch.zeros(idx.size(0), idx.size(0)).to(A.device)
        mask.fill_(float("0"))
        s1, t1 = A.topk(self._k, 1)
        mask.scatter_(1, t1, s1.fill_(1))
        A = A * mask
        return A
