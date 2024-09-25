import torch as th
import triton
import triton.language as tl
from torch import Tensor

__all__ = ["seqlen_to_index", "seqlen_to_batch_index"]


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
