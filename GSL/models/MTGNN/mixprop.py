import torch
import torch.nn as nn

from torch_geometric_temporal.nn.attention.mtgnn import Linear


class MixProp(nn.Module):
    r"""An implementation of the dynatic mix-hop propagation layer.
    For details see this paper: `"Connecting the Dots: Multivariate Time Series Forecasting with Graph Neural Networks."
    <https://arxiv.org/pdf/2005.11650.pdf>`_

    Args:
        c_in (int): Number of input channels.
        c_out (int): Number of output channels.
        gdep (int): Depth of graph convolution.
        dropout (float): Dropout rate.
        alpha (float): Ratio of retaining the root nodes's original states, a value between 0 and 1.
    """

    def __init__(self, c_in: int, c_out: int, gdep: int, alpha: float):
        super(MixProp, self).__init__()
        self._mlp = Linear((gdep + 1) * c_in, c_out)
        self._gdep = gdep
        self._alpha = alpha

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            else:
                nn.init.uniform_(p)

    def forward(self, X: torch.FloatTensor, A: torch.FloatTensor) -> torch.FloatTensor:
        """
        Making a forward pass of mix-hop propagation.

        Arg types:
            * **X** (Pytorch Float Tensor) - Input feature Tensor, with shape (c_in, num_nodes, seq_len).
            * **A** (PyTorch Float Tensor) - Adjacency matrix, with shape (num_nodes, num_nodes).

        Return types:
            * **H_0** (PyTorch Float Tensor) - Hidden representation for all nodes, with shape (c_out, num_nodes, seq_len).
        """
        A = A + torch.eye(A.size(0)).to(X.device)
        d = A.sum(1)
        H = X
        H_0 = X
        A = A / d.view(-1, 1)
        for _ in range(self._gdep):
            H = self._alpha * X + (1 - self._alpha) * torch.einsum(
                "cwl,vw->cvl", (H, A)
            )
            H_0 = torch.cat((H_0, H), dim=1)
        H_0 = self._mlp(H_0)
        return H_0