from functools import reduce
from typing import Any, Callable, List

import torch
import torch.distributed as dist
from torch import Tensor

__all__ = [
    "safe_all_reduce",
    "safe_all_mean",
    "safe_all_gather",
    "safe_all_gather_object",
    "safe_all_gather_list_object",
    "safe_barrier",
    "safe_broadcast",
    "is_rankzero",
    "rankzero_only",
]


def safe_all_reduce(x, reduce_op=dist.ReduceOp.SUM) -> torch.Tensor:
    if dist.is_initialized():
        dist.all_reduce(x, reduce_op)
    return x


def safe_all_mean(x) -> torch.Tensor:
    x = safe_all_reduce(x)
    if dist.is_initialized():
        x /= dist.get_world_size()
    return x


def safe_all_gather(x: Tensor, dim=0) -> torch.Tensor:
    if dist.is_initialized():
        xs = [torch.empty_like(x) for _ in range(dist.get_world_size())]
        dist.all_gather(xs, x)
        x = torch.cat(xs, dim=dim)
    return x


def safe_all_gather_object(x: Any) -> List[Any]:
    if dist.is_initialized():
        xs = [None] * dist.get_world_size()
        dist.all_gather_object(xs, x)
        x = xs
    return x


def safe_all_gather_list_object(x: List[Any]) -> List[Any]:
    if dist.is_initialized():
        xs = [None] * dist.get_world_size()
        dist.all_gather_object(xs, x)
        x = reduce(lambda a, b: a + b, xs, [])
    return x


def safe_barrier():
    if dist.is_initialized():
        dist.barrier()


def safe_broadcast(x, src) -> torch.Tensor:
    if dist.is_initialized():
        dist.broadcast(x, src)
    return x


def is_rankzero():
    if dist.is_initialized():
        return dist.get_rank() == 0
    else:
        return True


def rankzero_only(default=None):
    assert not isinstance(default, Callable)

    def decorator(func):
        def wrapper_func(*args, **kwargs):
            if is_rankzero():
                output = func(*args, **kwargs)
            else:
                output = default

            safe_barrier()
            return output

        return wrapper_func

    return decorator
