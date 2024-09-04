"""https://github.com/Kitsunetic/GEGLU-triton
MIT License

Copyright (c) 2024 Kitsunetic

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import math

import torch as th
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
from torch import Tensor
from torch.autograd.function import Function

__all__ = ["geglu", "GEGLU"]

_kAlpha = math.sqrt(2 / math.pi)


def cummul(*xs):
    y = xs[0]
    for x in xs[1:]:
        y *= x
    return y


@triton.jit
def tanh(x):
    # Tanh is just a scaled sigmoid
    return 2 * tl.sigmoid(2 * x) - 1


@triton.jit
def cosh(x):
    return (tl.exp(x) + tl.exp(-x)) * 0.5


@triton.jit
def gelu_forward(x):
    """
    GeLU_ activation - Gaussian error linear unit

    .. _GeLU: https://arxiv.org/pdf/1606.08415.pdf
    """
    return 0.5 * x * (1 + tanh(_kAlpha * x * (1 + 0.044715 * x * x)))


@triton.jit
def gelu_backward(x):
    x2 = x * x
    tanh_ = tanh(_kAlpha * x * (1 + 0.044715 * x2))
    dx = 0.5 * (x * (1 - tanh_ * tanh_) * (0.1070322244089 * x2 + 0.797884560802865) + tanh_ + 1)
    return dx


@triton.jit
def geglu_forward_kernel(x_ptr, y_ptr, N, C, C2, BLK_C: tl.constexpr, BLK_N: tl.constexpr):
    pid_n = tl.program_id(0)
    pid_c = tl.program_id(1)
    offs_n = pid_n * BLK_N + tl.arange(0, BLK_N)
    offs_c = pid_c * BLK_C + tl.arange(0, BLK_C)
    mask_n = offs_n < N
    mask_c = offs_c < C2
    mask = mask_n[:, None] & mask_c[None, :]

    x_ptrs = x_ptr + offs_n[:, None] * C + offs_c[None, :]
    x1 = tl.load(x_ptrs, mask=mask)
    x2 = tl.load(x_ptrs + C2, mask=mask)
    y = x1 * gelu_forward(x2)

    y_ptrs = y_ptr + offs_n[:, None] * C2 + offs_c[None, :]
    tl.store(y_ptrs, y, mask=mask)


@triton.jit
def geglu_backward_kernel(x_ptr, dx_ptr, dy_ptr, N, C, C2, BLK_C: tl.constexpr, BLK_N: tl.constexpr):
    pid_n = tl.program_id(0)
    pid_c = tl.program_id(1)
    offs_n = pid_n * BLK_N + tl.arange(0, BLK_N)
    offs_c = pid_c * BLK_C + tl.arange(0, BLK_C)
    mask_n = offs_n < N
    mask_c = offs_c < C2
    mask = mask_n[:, None] & mask_c[None, :]

    x_ptrs = x_ptr + offs_n[:, None] * C + offs_c[None, :]
    x1 = tl.load(x_ptrs, mask=mask)
    x2 = tl.load(x_ptrs + C2, mask=mask)

    dy_ptrs = dy_ptr + offs_n[:, None] * C2 + offs_c[None, :]
    dy = tl.load(dy_ptrs, mask=mask)

    # x * F.gelu(gates)
    dx1 = dy * gelu_forward(x2)
    dx2 = dy * x1

    # F.gelu(gates)
    dx2 *= gelu_backward(x2)

    dx_ptrs = dx_ptr + offs_n[:, None] * C + offs_c[None, :]
    tl.store(dx_ptrs, dx1, mask=mask)
    tl.store(dx_ptrs + C2, dx2, mask=mask)


class GEGLUFunction(Function):
    @staticmethod
    def forward(ctx, x: Tensor):
        """
        - x: ... c, contiguous
        """
        N, C = cummul(*x.shape[:-1]), x.size(-1)
        C2 = C >> 1
        y = x.new_empty(*x.shape[:-1], C2)

        BLK_C = max(8, min(1024, triton.next_power_of_2(C2)))
        BLK_N = max(1, 1024 // BLK_C)
        grid = lambda meta: (triton.cdiv(N, meta["BLK_N"]), triton.cdiv(C2, meta["BLK_C"]))
        geglu_forward_kernel[grid](x, y, N, C, C2, BLK_C=BLK_C, BLK_N=BLK_N)

        ctx.save_for_backward(x)
        return y

    @staticmethod
    def backward(ctx, dy: Tensor):
        """
        - dy: ... c // 2, contiguous
        """
        (x,) = ctx.saved_tensors  # ... c
        N, C = cummul(*x.shape[:-1]), x.size(-1)
        C2 = C >> 1
        dx = th.empty_like(x)  # ... c

        BLK_C = max(8, min(1024, triton.next_power_of_2(C2)))
        BLK_N = max(1, 1024 // BLK_C)
        grid = lambda meta: (triton.cdiv(N, meta["BLK_N"]), triton.cdiv(C2, meta["BLK_C"]))

        geglu_backward_kernel[grid](x, dx, dy, N, C, C2, BLK_C=BLK_C, BLK_N=BLK_N)
        return dx


def geglu(x: Tensor):
    """
    input:
    - x: ... c
    """
    C = x.size(-1)
    assert C & 0x01 == 0, x.shape

    if not x.is_contiguous():
        x = x.contiguous()

    return GEGLUFunction.apply(x)


class GEGLU(nn.Module):
    def forward(self, x: Tensor):
        return geglu(x)
