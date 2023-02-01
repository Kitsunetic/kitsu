from typing import Sequence

import torch as th


def discrete_grid_sample(p, x):
    """
    - p: b n 3
    - x: b d r r r
    """
    r = x.size(-1)
    p = p[..., 0] * r**2 + p[..., 0] * r + p[..., 2]
    p = p[:, None, :].repeat(1, x.size(1), 1)  # b d n
    y = x.flatten(2).gather(-1, p)  # b d n
    y = y.transpose_(1, 2).contiguous()  # b n d
    return y


def random_sample(x, n, dim=-1):
    idx = th.randperm(x.size(dim))[:n]
    if dim < 0:
        dim = x.dim() + dim
    u = [slice(None) for _ in range(dim)] + [idx]
    return x[u]


def batched_randperm(shape, dim=-1, device="cpu"):
    """adapted from https://discuss.pyth.org/t/batch-version-of-torch-randperm/111121/2"""
    idx = th.argsort(th.rand(shape, device=device), dim=dim)
    return idx


def unsqueeze_as(x, y) -> th.Tensor:
    if isinstance(y, th.Tensor):
        d = y.dim()
    else:
        d = len(y)
    return x.view(list(x.shape) + [1] * (d - x.dim()))


def cumprod(x: Sequence):
    out = 1
    for v in x:
        out *= v
    return out
