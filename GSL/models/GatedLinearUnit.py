import torch
import torch.nn as nn


class GLU(nn.Module):
    def __init__(self, input_length, hidden_dim):
        super(GLU, self).__init__()
        self.input_length = input_length
        self.hidden_dim = hidden_dim

        self.gated_linear_layer = nn.Linear(self.input_length, self.hidden_dim)
        self.message_layer = nn.Linear(self.input_length, self.hidden_dim)

    def forward(self, x):
        gate = torch.sigmoid(self.gated_linear_layer(x))
        message = self.message_layer(x)

        return torch.mul(gate, message)

