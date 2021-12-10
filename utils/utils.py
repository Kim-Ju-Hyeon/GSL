import numpy as np

def build_fully_connected_edge_idx(num_nodes):
    fully_connected = np.ones((num_nodes, num_nodes)) - np.eye(num_nodes)
    _edge = np.where(fully_connected)
    edge_index = np.array([_edge[0], _edge[1]], dtype=np.int64)
    return edge_index