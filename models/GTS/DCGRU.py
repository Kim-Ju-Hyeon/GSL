import math
import torch
from torch_geometric.utils import to_dense_adj, dense_to_sparse
from torch_geometric.nn.conv import MessagePassing
from torch_geometric_temporal.nn.recurrent.dcrnn import DConv

class DCRNN(torch.nn.Module):
    r"""An implementation of the Diffusion Convolutional Gated Recurrent Unit.
    For details see: `"Diffusion Convolutional Recurrent Neural Network:
    Data-Driven Traffic Forecasting" <https://arxiv.org/abs/1707.01926>`_

    Args:
        in_channels (int): Number of input features.
        out_channels (int): Number of output features.
        K (int): Filter size :math:`K`.
        bias (bool, optional): If set to :obj:`False`, the layer
            will not learn an additive bias (default :obj:`True`)

    """

    def __init__(self, in_channels: int, out_channels: int, K: int, bias: bool = True):
        super(DCRNN, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.K = K
        self.bias = bias

        self._create_parameters_and_layers()

    def _create_update_gate_parameters_and_layers(self):
        self.conv_x_z = DConv(
            in_channels=self.in_channels + self.out_channels,
            out_channels=self.out_channels,
            K=self.K,
            bias=self.bias,
        )

    def _create_reset_gate_parameters_and_layers(self):
        self.conv_x_r = DConv(
            in_channels=self.in_channels + self.out_channels,
            out_channels=self.out_channels,
            K=self.K,
            bias=self.bias,
        )

    def _create_candidate_state_parameters_and_layers(self):
        self.conv_x_h = DConv(
            in_channels=self.in_channels + self.out_channels,
            out_channels=self.out_channels,
            K=self.K,
            bias=self.bias,
        )

    def _create_parameters_and_layers(self):
        self._create_update_gate_parameters_and_layers()
        self._create_reset_gate_parameters_and_layers()
        self._create_candidate_state_parameters_and_layers()

    def _set_hidden_state(self, X, H):
        if H is None:
            H = torch.zeros(X.shape[0], self.out_channels).to(X.device)

        print(f"hidden state dim:{H.shape}")
        return H

    def _calculate_update_gate(self, X, edge_index, edge_weight, H):
        Z = torch.cat([X, H], dim=1)
        Z = self.conv_x_z(Z, edge_index, edge_weight)
        Z = torch.sigmoid(Z)
        return Z

    def _calculate_reset_gate(self, X, edge_index, edge_weight, H):
        R = torch.cat([X, H], dim=1)
        R = self.conv_x_r(R, edge_index, edge_weight)
        R = torch.sigmoid(R)
        return R

    def _calculate_candidate_state(self, X, edge_index, edge_weight, H, R):
        H_tilde = torch.cat([X, H * R], dim=1)
        H_tilde = self.conv_x_h(H_tilde, edge_index, edge_weight)
        H_tilde = torch.tanh(H_tilde)
        return H_tilde

    def _calculate_hidden_state(self, Z, H, H_tilde):
        H = Z * H + (1 - Z) * H_tilde
        return H

    def forward(
        self,
        X: torch.FloatTensor,
        edge_index: torch.LongTensor,
        edge_weight: torch.FloatTensor = None,
        H: torch.FloatTensor = None,
                                    ) -> torch.FloatTensor:
        r"""Making a forward pass. If edge weights are not present the forward pass
        defaults to an unweighted graph. If the hidden state matrix is not present
        when the forward pass is called it is initialized with zeros.

        Arg types:
            * **X** (PyTorch Float Tensor) - Node features.
            * **edge_index** (PyTorch Long Tensor) - Graph edge indices.
            * **edge_weight** (PyTorch Long Tensor, optional) - Edge weight vector.
            * **H** (PyTorch Float Tensor, optional) - Hidden state matrix for all nodes.

        Return types:
            * **H** (PyTorch Float Tensor) - Hidden state matrix for all nodes.
        """
        H = self._set_hidden_state(X, H)
        Z = self._calculate_update_gate(X, edge_index, edge_weight, H)
        R = self._calculate_reset_gate(X, edge_index, edge_weight, H)
        H_tilde = self._calculate_candidate_state(X, edge_index, edge_weight, H, R)
        H = self._calculate_hidden_state(Z, H, H_tilde)
        return H