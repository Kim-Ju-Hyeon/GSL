import torch
from torch.nn import functional as F
from torch_geometric.utils import to_dense_adj, dense_to_sparse, to_undirected

from utils.utils import build_batch_edge_index, build_batch_edge_weight, build_fully_connected_edge_idx


def gumbel_softmax_structure_sampling(theta, edge_index, batch_size: int, tau: float, node_nums: int, symmetric: bool):
    if theta.shape[0] == node_nums * node_nums:
        edge_probability = F.gumbel_softmax(theta, tau=tau, hard=True)
        connect = torch.where(edge_probability[:, 0])

        new_edge_index = torch.stack([edge_index[0, :][connect],
                                      edge_index[1, :][connect]])

    elif theta.shape[0] == node_nums:
        if symmetric:
            theta = (theta + theta.T) * 0.5
        edge_probability = F.gumbel_softmax(theta, tau=tau, hard=True)
        new_edge_index, _ = dense_to_sparse(edge_probability[:, :, 0])

    else:
        raise ValueError('Invalid gumbel softmax dim')

    batch_adj_matrix = build_batch_edge_index(new_edge_index, batch_size, node_nums)
    adj_matrix = to_dense_adj(new_edge_index).squeeze(dim=0)

    return batch_adj_matrix, adj_matrix


def weight_matrix_construct(theta, batch_size: int, node_nums: int, symmetric: bool):
    if theta.shape[0] == node_nums * node_nums:
        theta = theta.view(node_nums, node_nums)
    elif theta.shape[0] == node_nums:
        pass

    if symmetric:
        theta = (theta + theta.T) * 0.5

    # weight_matrix = theta
    # init_edge_index = build_fully_connected_edge_idx(node_nums)

    new_edge_index, weight_matrix = dense_to_sparse(theta)
    batch_adj_matrix = build_batch_edge_index(new_edge_index, batch_size, node_nums)
    batch_weight_matrix = build_batch_edge_weight(weight_matrix, batch_size)

    return batch_adj_matrix, batch_weight_matrix, theta


def top_k_structure_construct(theta, batch_size: int, k: int, node_nums: int, symmetric: bool, device):
    if theta.shape[0] == node_nums * node_nums:
        theta = theta.view(node_nums, node_nums)
    elif theta.shape[0] == node_nums:
        pass

    if symmetric:
        theta = (theta + theta.T) * 0.5

    mask = torch.zeros(node_nums, node_nums).to(device=device)

    mask.fill_(float("0"))
    s1, t1 = theta.topk(k, 1)
    mask.scatter_(1, t1, s1.fill_(1))
    adj_matrix = theta * mask
    new_edge_index, new_weight_index = dense_to_sparse(adj_matrix)

    batch_adj_matrix = build_batch_edge_index(new_edge_index, batch_size, node_nums)
    batch_weight_matrix = build_batch_edge_weight(new_weight_index, batch_size)

    return batch_adj_matrix, adj_matrix
