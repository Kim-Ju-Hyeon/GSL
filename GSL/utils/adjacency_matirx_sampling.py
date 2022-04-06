import torch
import torch.nn as nn
from torch.nn import functional as F
from torch_geometric.utils import to_dense_adj, dense_to_sparse

from utils.utils import build_batch_edge_index, build_batch_edge_weight


def gumbel_softmax_structure_sampling(theta, edge_index, batch_size, tau, node_nums):
    if len(theta.size()) == 2:
        pass
    elif len(theta.size()) == 3:
        theta = theta.view(-1, 2)

    edge_probability = F.gumbel_softmax(theta, tau=tau, hard=True)
    connect = torch.where(edge_probability[:, 0])

    adj_matrix = torch.stack([edge_index[0, :][connect],
                              edge_index[1, :][connect]])

    batch_adj_matrix = build_batch_edge_index(adj_matrix, batch_size, node_nums)

    return batch_adj_matrix, adj_matrix


def weight_matrix_construct(theta, init_edge_index, batch_size, node_nums):
    if len(theta.size()) == 2:
        theta = theta.view(node_nums, node_nums)
    elif len(theta.size()) == 3:
        pass

    batch_adj_matrix = build_batch_edge_index(init_edge_index, batch_size, node_nums)
    batch_weight_matrix = build_batch_edge_weight(theta, batch_size)

    return batch_adj_matrix, batch_weight_matrix, theta


def top_k_structure_construct(theta, alpha, k, node_nums):
    if len(theta.size()) == 2:
        theta = theta.view(node_nums, node_nums)
    elif len(theta.size()) == 3:
        theta = theta.unsqueeze(dim=-1)

    A = F.relu(torch.tanh(alpha * theta))
    mask = torch.zeros(node_nums, node_nums)
    mask.fill_(float("0"))
    s1, t1 = A.topk(k, 1)
    mask.scatter_(1, t1, s1.fill_(1))
    A = A * mask

    return A