import numpy as np
import torch

def build_fully_connected_edge_idx(num_nodes):
    fully_connected = np.ones((num_nodes, num_nodes)) - np.eye(num_nodes)
    _edge = np.where(fully_connected)
    edge_index = np.array([_edge[0], _edge[1]], dtype=np.int64)
    return torch.LongTensor(edge_index)


def build_batch_edge_index(edge_index, num_graphs):
    new_edge = edge_index
    for num in range(1, num_graphs):
        next_graph_edge = edge_index + num * 100
        new_edge = torch.cat([new_edge, next_graph_edge], dim=-1)
    return new_edge