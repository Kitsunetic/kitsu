import random
from typing import Tuple

import torch as th
import triton
import triton.language as tl
from torch import Tensor

__all__ = ["seqlen_to_index", "seqlen_to_batch_index", "padding_index"]


@triton.jit
def seqlen_to_index_kernel(seqlen_ptr, idx_ptr, BLK: tl.constexpr):
    pid = tl.program_id(0)
    i = tl.load(seqlen_ptr + pid)
    j = tl.load(seqlen_ptr + pid + 1)
    idx = tl.arange(0, BLK)
    tl.store(idx_ptr + i + idx, idx, mask=idx < (j - i))


def seqlen_to_index(seqlen: Tensor, max_seqlen: int):
    """Conver seqlen into index.
    For example, seqlen = [0, 3, 5], then this function returns [0, 1, 2, 0, 1]

    Args:
        seqlen (Tensor): (batch_size + 1,), int32.
        max_seqlen (int): maximum sequence length (same with `(seqlen[1:]-seqlen[:-1]).max()`)
    Return:
        index (Tensor): (total,), int64
    """
    assert seqlen[0].item() == 0

    B = seqlen.size(0) - 1
    idx = seqlen.new_empty(seqlen[-1].item(), dtype=th.int64)
    BLK = triton.next_power_of_2(max_seqlen)
    seqlen_to_index_kernel[(B,)](seqlen, idx, BLK)
    return idx


@triton.jit
def seqlen_to_batch_index_kernel(seqlen_ptr, idx_ptr, BLK: tl.constexpr):
    pid = tl.program_id(0)
    i = tl.load(seqlen_ptr + pid)
    j = tl.load(seqlen_ptr + pid + 1)
    idx = tl.arange(0, BLK)
    tl.store(idx_ptr + i + idx, pid, mask=idx < (j - i))


def seqlen_to_batch_index(seqlen: Tensor, max_seqlen: int):
    """Convert seqlen into batch index.
    Fro example, seqlen = [0, 3, 5], then this function returns [0, 0, 0, 1, 1]

    Args:
        seqlen (Tensor): (batch_size + 1,), int32.
        max_seqlen (int): maximum sequence length (same with `(seqlen[1:]-seqlen[:-1]).max()`)
    Return:
        index (Tensor): (total,), int64
    """
    assert seqlen[0].item() == 0

    B = seqlen.size(0) - 1
    idx = seqlen.new_empty(seqlen[-1].item(), dtype=th.int64)
    BLK = triton.next_power_of_2(max_seqlen)
    seqlen_to_batch_index_kernel[(B,)](seqlen, idx, BLK)
    return idx


@triton.jit
def padding_index_kernel(seqlen_ptr, new_seqlen_ptr, new_max_seqlen, idx_ptr, window_size, seed, BLK_N: tl.constexpr):
    pid_b = tl.program_id(0)

    i1 = tl.load(seqlen_ptr + pid_b)
    j1 = tl.load(seqlen_ptr + pid_b + 1)
    i2 = tl.load(new_seqlen_ptr + pid_b)
    j2 = tl.load(new_seqlen_ptr + pid_b + 1)
    l1, l2 = j1 - i1, j2 - i2
    rnd_range_min = l1 - window_size
    rnd_range_min = tl.where(rnd_range_min < 0, 0, rnd_range_min)
    rnd_range = l1 - rnd_range_min

    for pid_n in range(tl.cdiv(new_max_seqlen, BLK_N)):
        idx = pid_n * BLK_N + tl.arange(0, BLK_N)  # n
        rnd = rnd_range_min + (tl.rand(seed, idx) * rnd_range).to(tl.int32)  # n, [0, l1 - 1]
        val = i1 + tl.where(idx >= l1, rnd, idx)
        tl.store(idx_ptr + i2 + idx, val.to(tl.int64), mask=idx < l2)


def padding_index(seqlen: Tensor, window_size: int) -> Tensor:
    """
    Args:
        seqlen (Tensor): (batch_size + 1,), int32.
        window_size (int):
    Returns:
        idx (Tensor): (M, ), int64
    """
    B = seqlen.size(0) - 1
    seed = int(random.random() * 1e6)

    pad_size = ((seqlen[:-1] - seqlen[1:]) % window_size).cumsum_(0)
    new_seqlen = seqlen.clone()
    new_seqlen[1:] += pad_size
    new_max_seqlen = (new_seqlen[1:] - new_seqlen[:-1]).amax().item()
    new_N = new_seqlen[-1].item()
    idx = th.empty(new_N, dtype=th.int64, device=seqlen.device)

    BLK_N = min(32, max(2048, triton.next_power_of_2(new_max_seqlen)))
    grid = (B,)
    padding_index_kernel[grid](seqlen, new_seqlen, new_max_seqlen, idx, window_size, seed, BLK_N=BLK_N)
    return idx, new_seqlen, new_max_seqlen
