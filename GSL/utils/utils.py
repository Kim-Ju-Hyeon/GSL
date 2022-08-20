import numpy as np
import torch
from torch_geometric.utils import add_self_loops, sort_edge_index


def build_fully_connected_edge_idx(num_nodes):
    fully_connected = np.ones((num_nodes, num_nodes)) - np.eye(num_nodes)
    _edge = np.where(fully_connected)
    edge_index = np.array([_edge[0], _edge[1]], dtype=np.int64)
    edge_index = sort_edge_index(add_self_loops(torch.LongTensor(edge_index))[0])
    return edge_index


def build_batch_edge_index(edge_index, num_graphs, num_nodes):
    new_edge = edge_index
    for num in range(1, num_graphs):
        next_graph_edge = edge_index + num * num_nodes
        new_edge = torch.cat([new_edge, next_graph_edge], dim=-1)
    return new_edge


def build_batch_edge_weight(edge_weight, num_graphs):
    out_list = []
    for _ in range(num_graphs):
        out_list.append(edge_weight)
    output = torch.stack(out_list).view(-1, 1)
    return output


def build_dynamic_batch_edge_index(edge_index):
    new_edge = edge_index[0]
    num = 1

    if len(edge_index) != 1:
        for batch_edge_index in edge_index[1:]:
            next_graph_edge = batch_edge_index + num * 98
            new_edge = torch.cat([new_edge, next_graph_edge], dim=-1)

            num += 1

        return new_edge

    else:
        return edge_index[0]

def squeeze_dim(tensor):
    if len(tensor.shape) == 3 and tensor.shape[-1] == 1:  # (128, 10, 1) => (128, 10).
        return tensor.squeeze()
    return tensor

