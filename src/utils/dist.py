import torch
import torch.distributed as dist


def safe_all_reduce(x, reduce_op=dist.ReduceOp.SUM):
    if dist.is_initialized():
        dist.all_reduce(x, reduce_op)
    return x


def safe_all_mean(x):
    x = safe_all_reduce(x)
    if dist.is_initialized():
        x /= dist.get_world_size()
    return x


def safe_all_gather(x, dim=0):
    if dist.is_initialized():
        xs = [torch.empty_like(x) for _ in range(dist.get_world_size())]
        dist.all_gather(xs, x)
        x = torch.cat(xs, dim=dim)
    return x


def safe_barrier():
    if dist.is_initialized():
        dist.barrier()


def safe_broadcast(x, src):
    if dist.is_initialized():
        dist.broadcast(x, src)
