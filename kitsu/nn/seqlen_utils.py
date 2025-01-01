from dataclasses import dataclass
from typing import Union

import torch as th
import triton
import triton.language as tl
from torch import Tensor
from torch.autograd import Function

__all__ = [
    "AttentionBatch",
    "seqlen_to_index",
    "seqlen_to_batch_index",
    "padding_index",
    "code_to_seqlen",
    "code_downscale",
]

MAX_THREADS = 4096
MIN_THREADS = 32


@dataclass
class AttentionBatch:
    x: Tensor
    seqlen: Tensor
    max_seqlen: int

    def new(self, x: Tensor = None, seqlen: Tensor = None, max_seqlen: int = None):
        return AttentionBatch(
            x=x if x is not None else self.x,
            seqlen=seqlen if seqlen is not None else self.seqlen,
            max_seqlen=max_seqlen if max_seqlen is not None else self.max_seqlen,
        )


@triton.jit
def clamp(x, amin, amax):
    x = tl.where(x < amin, amin, x)
    x = tl.where(x >= amax, amax, x)
    return x


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
def padding_index_kernel(seqlen_ptr, new_seqlen_ptr, new_max_seqlen, idx_ptr, window_size, BLK_N: tl.constexpr):
    pid_b = tl.program_id(0)
    i1 = tl.load(seqlen_ptr + pid_b).to(tl.int32)
    j1 = tl.load(seqlen_ptr + pid_b + 1).to(tl.int32)
    i2 = tl.load(new_seqlen_ptr + pid_b).to(tl.int32)
    j2 = tl.load(new_seqlen_ptr + pid_b + 1).to(tl.int32)

    for pid_n in range(tl.cdiv(new_max_seqlen, BLK_N)):
        offs_idx = pid_n * BLK_N + tl.arange(0, BLK_N)
        mask_idx = offs_idx < j2 - i2
        idx_ptrs = idx_ptr + i2 + offs_idx

        # padding
        idx = i1 + offs_idx.to(tl.int32)
        tmp = clamp(idx - window_size, i1, j1 - 1)
        idx_out = tl.where(idx < j1, idx, tmp)
        tl.store(idx_ptrs, idx_out, mask=mask_idx)


def padding_index(seqlen: Tensor, window_size: int) -> Tensor:
    """Giving padding so that the N can be dividable by window_size.
    Args:
        seqlen (Tensor): (batch_size + 1,), int32.
        window_size (int):
    Returns:
        idx (Tensor): (M, ), int64
    """
    B = seqlen.size(0) - 1

    pad_size = ((seqlen[:-1] - seqlen[1:]) % window_size).cumsum_(0)
    new_seqlen = seqlen.clone()
    new_seqlen[1:] += pad_size
    new_max_seqlen = (new_seqlen[1:] - new_seqlen[:-1]).amax().item()
    new_N = new_seqlen[-1].item()
    idx = th.empty(new_N, dtype=th.int64, device=seqlen.device)

    BLK_N = max(MIN_THREADS, min(MAX_THREADS, triton.next_power_of_2(new_max_seqlen)))
    grid = (B,)
    padding_index_kernel[grid](seqlen, new_seqlen, new_max_seqlen, idx, window_size, BLK_N=BLK_N)
    return idx, new_seqlen, new_max_seqlen


@triton.jit
def code_to_seqlen_kernel(code_ptr, seqlen_ptr, B, N, BLK: tl.constexpr):
    pid = tl.program_id(0)
    out = tl.zeros((1,), dtype=tl.int32)

    for nidx in range(tl.cdiv(N, BLK)):
        offs_n = nidx * BLK + tl.arange(0, BLK)
        mask_n = offs_n < N
        code = tl.load(code_ptr + offs_n, mask=mask_n, other=0x7FFF << 48)

        bidx = ((code >> 48) & 0x7FFF).to(tl.int32)
        x = tl.min((bidx == pid).to(tl.int32) * (offs_n - 0xFFFF), axis=0)
        out = tl.minimum(out, x)

    out = tl.where(out == 0, -1, out + 0xFFFF)
    tl.store(seqlen_ptr + pid + tl.arange(0, 1), out)

    # set right-side value
    tl.store(seqlen_ptr + B, N, mask=pid == B - 1)


def code_to_seqlen(code: Tensor, batch_size: int) -> Tensor:
    """Convert code to seqlen.
    Args:
        code (Tensor): N, int64
    Returns:
        seqlen (Tensor): (batch_size+1, ), int32
    """
    # top 16 bits are allocated for batch index
    B, N = batch_size, len(code)
    seqlen = code.new_empty(batch_size + 1, dtype=th.int32)
    BLK = max(MIN_THREADS, min(MIN_THREADS, triton.next_power_of_2(N)))
    grid = (B,)
    code_to_seqlen_kernel[grid](code, seqlen, B, N, BLK)
    max_seqlen = (seqlen[1:] - seqlen[:-1]).amax().item()
    return seqlen, max_seqlen


@triton.jit
def code_downscale_kernel(code_ptr, out_ptr, n_steps, N, BLK: tl.constexpr):
    pid = tl.program_id(0)
    offs_n = BLK * pid + tl.arange(0, BLK)
    mask_n = offs_n < N
    code = tl.load(code_ptr + offs_n, mask=mask_n)

    top16bit = code & (0x7FFF << 48)
    low16bit = code & ((1 << 48) - 1)
    low16bit >>= n_steps * 3

    new_code = top16bit | low16bit
    tl.store(out_ptr + offs_n, new_code, mask=mask_n)


def code_downscale(code: Tensor, n_steps: int):
    assert code.ndim == 1 and code.dtype == th.int64, f"{code.shape}, {code.dtype}"
    N = len(code)
    new_code = th.empty_like(code)
    BLK = max(MIN_THREADS, min(MAX_THREADS, triton.next_power_of_2(N)))
    grid = (triton.cdiv(N, BLK),)
    code_downscale_kernel[grid](code, new_code, n_steps, N, BLK)
    return new_code


@triton.jit
def padded_batch_fwd_kernel(x_ptr, seqlen_ptr, out_ptr, other, M, C, BLK_C: tl.constexpr, BLK_M: tl.constexpr):
    """
    Args:
        x: N c
        out: b m c
    """
    bidx = tl.program_id(0)
    midx = tl.program_id(1)

    offs_m = midx * BLK_M + tl.arange(0, BLK_M)

    i = tl.load(seqlen_ptr + bidx)
    j = tl.load(seqlen_ptr + bidx + 1)

    for cidx in range(tl.cdiv(C, BLK_C)):
        offs_c = cidx * BLK_C + tl.arange(0, BLK_C)
        x_ptrs = x_ptr + i * C + offs_m[None, :] * C + offs_c[:, None]  # m c
        x_mask = ((i + offs_m[None, :]) < j) & (offs_c[:, None] < C)

        out_ptrs = out_ptr + bidx * M * C + offs_m[None, :] * C + offs_c[:, None]  # m c
        out_mask = (offs_m[None, :] < M) & (offs_c[:, None] < C)

        x = tl.load(x_ptrs, x_mask, other)
        tl.store(out_ptrs, x, out_mask)


@triton.jit
def padded_batch_bwd_kernel(x_grad_ptr, seqlen_ptr, out_grad_ptr, M, C, BLK_C: tl.constexpr, BLK_M: tl.constexpr):
    """
    Args:
        x_grad_ptr: N c
        out_grad_ptr: b m c
    """
    bidx = tl.program_id(0)
    midx = tl.program_id(1)

    offs_m = midx * BLK_M + tl.arange(0, BLK_M)

    i = tl.load(seqlen_ptr + bidx)
    j = tl.load(seqlen_ptr + bidx + 1)

    for cidx in range(tl.cdiv(C, BLK_C)):
        offs_c = cidx * BLK_C + tl.arange(0, BLK_C)
        x_grad_ptrs = x_grad_ptr + i * C + offs_m[None, :] * C + offs_c[:, None]  # m c
        x_grad_mask = (i + offs_m[None, :] < j) & (offs_c[:, None] < C)

        out_grad_ptrs = out_grad_ptr + bidx * M * C + offs_m[None, :] * C + offs_c[:, None]  # m c
        out_grad_mask = (offs_m[None, :] < M) & (offs_c[:, None] < C)

        out_grad = tl.load(out_grad_ptrs, out_grad_mask)
        tl.store(x_grad_ptrs, out_grad, x_grad_mask)


class PaddedBatch(Function):
    @staticmethod
    def forward(ctx, x: Tensor, seqlen: Tensor, max_seqlen: Tensor, other=0.0):
        assert x.is_contiguous()

        B, M, C = len(seqlen) - 1, *x.shape
        out = x.new_empty(B, max_seqlen, C)

        BLK_C = min(MAX_THREADS, max(MIN_THREADS, triton.next_power_of_2(C)))
        BLK_M = min(MAX_THREADS // BLK_C, triton.next_power_of_2(M))
        GRID_M = triton.cdiv(M, BLK_M)
        padded_batch_fwd_kernel[(B, GRID_M)](x, seqlen, out, other, M, C, BLK_C, BLK_M)

        ctx.save_for_backward(seqlen)
        return out

    @staticmethod
    def backward(ctx, grad_out: Tensor):
        (seqlen,) = ctx.saved_tensors
        B, M, C, N = *grad_out.shape, seqlen[-1].item()
        grad_x = grad_out.new_empty(N, C)

        BLK_C = min(MAX_THREADS, max(MIN_THREADS, triton.next_power_of_2(C)))
        BLK_M = min(MAX_THREADS // BLK_C, triton.next_power_of_2(M))
        GRID_M = triton.cdiv(M, BLK_M)
        padded_batch_bwd_kernel[(B, GRID_M)](grad_x, seqlen, grad_out, M, C, BLK_C, BLK_M)
        return grad_x


def padded_batch(x: Union[Tensor, AttentionBatch], seqlen: Tensor = None, max_seqlen: int = None):
    if isinstance(x, AttentionBatch):
        assert seqlen is None and max_seqlen is None
        return PaddedBatch.apply(x.x, x.seqlen, x.max_seqlen)
    elif isinstance(x, Tensor):
        assert seqlen is not None and max_seqlen is not None
        return PaddedBatch.apply(x, seqlen, max_seqlen)
    else:
        raise NotImplementedError(str(type(x)))
