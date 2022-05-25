import torch
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree


class MPNN(MessagePassing):
    def __init__(self):
        super(MPNN, self).__init__()


