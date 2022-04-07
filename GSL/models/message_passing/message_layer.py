from typing import List, Optional, Set, Callable, get_type_hints
from torch_geometric.typing import Adj, Size

import os
import re
import inspect
import os.path as osp
from uuid import uuid1
from itertools import chain
from inspect import Parameter
from collections import OrderedDict

import torch
from torch import Tensor
from jinja2 import Template
from torch.utils.hooks import RemovableHandle
from torch_sparse import SparseTensor
from torch_scatter import gather_csr, scatter, segment_csr
from torch_geometric.nn import MessagePassing


class MessageLayer(MessagePassing):
    def __init__(self):
        super(MessageLayer, self).__init__()
        self.total_message = None
        self.__explain__ = False

    def propagate(self, edge_index: Adj, size: Size = None, **kwargs):
        r"""The initial call to start propagating messages.
        Args:
            edge_index (Tensor or SparseTensor): A :obj:`torch.LongTensor` or a
                :obj:`torch_sparse.SparseTensor` that defines the underlying
                graph connectivity/message passing flow.
                :obj:`edge_index` holds the indices of a general (sparse)
                assignment matrix of shape :obj:`[N, M]`.
                If :obj:`edge_index` is of type :obj:`torch.LongTensor`, its
                shape must be defined as :obj:`[2, num_messages]`, where
                messages from nodes in :obj:`edge_index[0]` are sent to
                nodes in :obj:`edge_index[1]`
                (in case :obj:`flow="source_to_target"`).
                If :obj:`edge_index` is of type
                :obj:`torch_sparse.SparseTensor`, its sparse indices
                :obj:`(row, col)` should relate to :obj:`row = edge_index[1]`
                and :obj:`col = edge_index[0]`.
                The major difference between both formats is that we need to
                input the *transposed* sparse adjacency matrix into
                :func:`propagate`.
            size (tuple, optional): The size :obj:`(N, M)` of the assignment
                matrix in case :obj:`edge_index` is a :obj:`LongTensor`.
                If set to :obj:`None`, the size will be automatically inferred
                and assumed to be quadratic.
                This argument is ignored in case :obj:`edge_index` is a
                :obj:`torch_sparse.SparseTensor`. (default: :obj:`None`)
            **kwargs: Any additional data which is needed to construct and
                aggregate messages, and to update node embeddings.
        """

        decomposed_layers = 1   # if self.__explain__ else self.decomposed_layers

        for hook in self._propagate_forward_pre_hooks.values():
            res = hook(self, (edge_index, size, kwargs))
            if res is not None:
                edge_index, size, kwargs = res

        size = self.__check_input__(edge_index, size)

        # Run "fused" message and aggregation (if applicable).
        if (isinstance(edge_index, SparseTensor) and self.fuse
                and not self.__explain__):
            coll_dict = self.__collect__(self.__fused_user_args__, edge_index,
                                         size, kwargs)

            msg_aggr_kwargs = self.inspector.distribute(
                'message_and_aggregate', coll_dict)

            for hook in self._message_and_aggregate_forward_pre_hooks.values():
                res = hook(self, (edge_index, msg_aggr_kwargs))
                if res is not None:
                    edge_index, msg_aggr_kwargs = res
            out = self.message_and_aggregate(edge_index, **msg_aggr_kwargs)

            for hook in self._message_and_aggregate_forward_hooks.values():
                res = hook(self, (edge_index, msg_aggr_kwargs), out)
                if res is not None:
                    out = res

            update_kwargs = self.inspector.distribute('update', coll_dict)
            out = self.update(out, **update_kwargs)

        # Otherwise, run both functions in separation.
        elif isinstance(edge_index, Tensor) or not self.fuse:
            if decomposed_layers > 1:
                user_args = self.__user_args__
                decomp_args = {a[:-2] for a in user_args if a[-2:] == '_j'}
                decomp_kwargs = {
                    a: kwargs[a].chunk(decomposed_layers, -1)
                    for a in decomp_args
                }
                decomp_out = []

            for i in range(decomposed_layers):
                if decomposed_layers > 1:
                    for arg in decomp_args:
                        kwargs[arg] = decomp_kwargs[arg][i]

                coll_dict = self.__collect__(self.__user_args__, edge_index,
                                             size, kwargs)

                msg_kwargs = self.inspector.distribute('message', coll_dict)

                for hook in self._message_forward_pre_hooks.values():
                    res = hook(self, (msg_kwargs,))
                    if res is not None:
                        msg_kwargs = res[0] if isinstance(res, tuple) else res
                out = self.message(**msg_kwargs)
                self.total_message = out

                for hook in self._message_forward_hooks.values():
                    res = hook(self, (msg_kwargs,), out)
                    if res is not None:
                        out = res

                # For `GNNExplainer`, we require a separate message and
                # aggregate procedure since this allows us to inject the
                # `edge_mask` into the message passing computation scheme.
                if self.__explain__:
                    edge_mask = self.__edge_mask__.sigmoid()
                    # Some ops add self-loops to `edge_index`. We need to do
                    # the same for `edge_mask` (but do not train those).
                    if out.size(self.node_dim) != edge_mask.size(0):
                        edge_mask = edge_mask[self.__loop_mask__]
                        loop = edge_mask.new_ones(size[0])
                        edge_mask = torch.cat([edge_mask, loop], dim=0)
                    assert out.size(self.node_dim) == edge_mask.size(0)
                    out = out * edge_mask.view([-1] + [1] * (out.dim() - 1))

                aggr_kwargs = self.inspector.distribute('aggregate', coll_dict)

                for hook in self._aggregate_forward_pre_hooks.values():
                    res = hook(self, (aggr_kwargs,))
                    if res is not None:
                        aggr_kwargs = res[0] if isinstance(res, tuple) else res
                out = self.aggregate(out, **aggr_kwargs)

                for hook in self._aggregate_forward_hooks.values():
                    res = hook(self, (aggr_kwargs,), out)
                    if res is not None:
                        out = res

                update_kwargs = self.inspector.distribute('update', coll_dict)
                out = self.update(out, **update_kwargs)

                if decomposed_layers > 1:
                    decomp_out.append(out)

            if decomposed_layers > 1:
                out = torch.cat(decomp_out, dim=-1)

        for hook in self._propagate_forward_hooks.values():
            res = hook(self, (edge_index, size, kwargs), out)
            if res is not None:
                out = res

        return out, self.total_message
