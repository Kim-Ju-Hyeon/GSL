import torch
import torch.nn.functional as F
import torch.nn as nn

from torch_geometric.nn import GCNConv

class GCN(torch.nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.Inter_Series_Stack = nn.ModuleList()
        for i in range(len(channels)):
            self.Inter_Series_Stack.append(GCNConv(channels[i], channels[i+1]))
            self.Inter_Series_Stack.append(nn.ReLU())

    def forward(self, x, edge_index):
        x = self.Inter_Series_Stack(x, edge_index)
        return x