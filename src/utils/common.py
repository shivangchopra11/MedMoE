import warnings
from collections import OrderedDict
from copy import deepcopy
from dataclasses import fields
from typing import Any, Callable, List, Tuple, Union

import torch
from torch import nn, Tensor
from torch.utils.checkpoint import checkpoint


def shift_dim(
    x: Tensor, src_dim: int = -1, dest_dim: int = -1, make_contiguous: bool = True
) -> Tensor:
    """Permutes tensor x by moving src_dim to dest_dim.
    i.e. shift_dim(x, 1, -1) would be (b, c, t, h, w) -> (b, t, h, w, c)

    Code taken from VideoGPT
    https://github.com/wilson1yan/VideoGPT/blob/master/videogpt/utils.py

    Args:
        x (Tensor): input Tensor you want to permute
        src_dim (int, optional): the axis you want to move. Negative indexing supported. Defaults to -1.
        dest_dim (int, optional): the axis you want to move to. Negative indexing supported. Defaults to -1.
        make_contiguous (bool, optional): if you want the output tensor to be contiguous in memory. Defaults to True.

    Returns:
        Tensor: permuted Tensor
    """
    n_dims = len(x.shape)
    # Remap negative dim
    if src_dim < 0:
        src_dim = n_dims + src_dim
    if dest_dim < 0:
        dest_dim = n_dims + dest_dim

    assert 0 <= src_dim < n_dims and 0 <= dest_dim < n_dims

    dims = [i for i in range(n_dims) if i != src_dim]

    permutation = []
    ctr = 0
    for i in range(n_dims):
        if i == dest_dim:
            permutation.append(src_dim)
        else:
            permutation.append(dims[ctr])
            ctr += 1
    x = x.permute(permutation)
    if make_contiguous:
        x = x.contiguous()
    return x

class ModelOutput(OrderedDict):
    def keys(self) -> Any:
        for field in fields(self):  # type: ignore
            yield field.name

    def __getitem__(self, key: Any) -> Any:
        return getattr(self, key)

    def __iter__(self) -> Any:
        yield from self.keys()

    def values(self) -> Any:
        for field in fields(self):  # type: ignore
            yield getattr(self, field.name)

    def items(self) -> Any:
        for field in fields(self):  # type: ignore
            yield field.name, getattr(self, field.name)