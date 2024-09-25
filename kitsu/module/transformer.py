"""
flash-attn (v1, no flash-attn v2 or n3 for RTX 2080Ti compatibility) based general purpose transformer architecture.

@kitsunetic
"""

from functools import partial
from pdb import set_trace

import torch as th
import torch.nn as nn
import torch.nn.init as init
import triton
from einops import rearrange, repeat
from flash_attn.flash_attn_interface import flash_attn_unpadded_func
from torch import Tensor

from kitsu.module import GEGLU, seqlen_to_index

__all__ = ["TransformerLayer", "TransformerBlock", "TransformerBlockBatched"]


class RoPEUnpadded(nn.Module):
    # TODO kernel fusion (qkv-packed) & float16 output
    def __init__(self, dim: int, scale=10000):
        super().__init__()
        self._freqs_core = 1.0 / (scale ** (th.arange(0, dim, 2, dtype=th.float) / dim))  # c/2
        self._N = 0

    def get_freqs(self, x: Tensor, seqlen: Tensor, max_seqlen: int):
        """Cache frequency"""
        N = max_seqlen
        if N > self._N:
            self._N = triton.next_power_of_2(N)
            freqs_core = self._freqs_core.to(x.device)

            pos = th.arange(self._N, dtype=x.dtype, device=x.device)  # m
            freqs = pos[:, None] * freqs_core[None, :].type_as(x)  # m c/2
            freqs = repeat(freqs, "m c -> m 1 (c x)", x=2).contiguous()  # m 1 c
            self._freqs_cos = freqs.cos()
            self._freqs_sin = freqs.sin()

        idx = seqlen_to_index(seqlen, max_seqlen)  # (total,), int64

        freqs_cos = self._freqs_cos[idx]  # total 1 c
        freqs_sin = self._freqs_sin[idx]  # total 1 c
        return freqs_cos, freqs_sin

    def forward(self, x: Tensor, seqlen: Tensor, max_seqlen: int):
        """
        - input x: (total h c)
        - input seqlen: (batch_size + 1), int32
        - input max_seqlen: int.
        - output: (total h c)
        """
        freqs_cos, freqs_sin = self.get_freqs(x, seqlen, max_seqlen)

        x1, x2 = rearrange(x, "o h (c x) -> o h c x", x=2).unbind(-1)
        x_rot = th.stack([-x2, x1], -1).flatten(-2)  # total h c/2 2 -> total h c
        out = x.contiguous() * freqs_cos.contiguous() + x_rot.contiguous() * freqs_sin.contiguous()
        return out


class FFN(nn.Sequential):
    def __init__(self, dim, expansion=4.0):
        super().__init__(
            nn.LayerNorm(dim, eps=1e-6),
            nn.Linear(dim, int(dim * expansion) * 2),
            GEGLU(),
            nn.Linear(int(dim * expansion), dim),
        )

        self.reset_parameters()

    def reset_parameters(self):
        init.trunc_normal_(self[1].weight, std=0.02)
        init.trunc_normal_(self[3].weight, std=0.02)
        init.zeros_(self[1].bias)
        init.zeros_(self[3].bias)


class TransformerLayer(nn.Module):
    def __init__(self, dim: int, dim_context: int = None, head_dim=64, dropout=0.0, expansion=4.0, causal=False):
        assert dim % head_dim == 0, (dim, head_dim)
        dim_context = dim_context or dim
        super().__init__()
        self.head_dim = head_dim
        self.dropout = dropout
        self.causal = causal

        self.pe = RoPEUnpadded(head_dim)
        self.norm_q = nn.LayerNorm(dim, eps=1e-6)
        self.to_q = nn.Linear(dim, dim, bias=False)
        self.norm_kv = nn.LayerNorm(dim_context, eps=1e-6)
        self.to_k = nn.Linear(dim_context, dim, bias=False)
        self.to_v = nn.Linear(dim_context, dim, bias=False)
        self.to_out = nn.Linear(dim, dim)
        self.ffn = FFN(dim, expansion=expansion)

        self.reset_parameters()

    def reset_parameters(self):
        init.trunc_normal_(self.to_q.weight, std=0.02)
        init.trunc_normal_(self.to_k.weight, std=0.02)
        init.trunc_normal_(self.to_v.weight, std=0.02)
        init.trunc_normal_(self.to_out.weight, std=0.02)
        init.zeros_(self.to_out.bias)

    def forward(
        self,
        x: Tensor,
        seqlen: Tensor,
        max_seqlen: int,
        context: Tensor = None,
        seqlen_kv: Tensor = None,
        max_seqlen_kv: int = None,
    ):
        """
        - input x:                  total c
        - input seqlen:             total + 1, int32
        - input max_seqlen:         int, longest batch's length in query
        - input [context]:          total' dim_context
        - input [seqlen_kv]:        total' + 1, int32
        - input [max_seqlen_kv]:    int, longest batch's length in key-value
        - output:                   total c
        """
        assert (context is None) == (seqlen_kv is None) == (max_seqlen_kv is None)
        if context is None:
            context, seqlen_kv, max_seqlen_kv = x, seqlen, max_seqlen

        q = self.to_q(self.norm_q(x))
        context = self.norm_kv(context)
        k = self.to_k(context)
        v = self.to_v(context)
        q = rearrange(q, "t (h c) -> t h c", c=self.head_dim)
        k = rearrange(k, "t (h c) -> t h c", c=self.head_dim)
        v = rearrange(v, "t (h c) -> t h c", c=self.head_dim)
        q = self.pe(q, seqlen, max_seqlen)
        k = self.pe(k, seqlen_kv, max_seqlen_kv)
        out = flash_attn_unpadded_func(
            q=q.half(),
            k=k.half(),
            v=v.half(),
            cu_seqlens_q=seqlen,
            cu_seqlens_k=seqlen_kv,
            max_seqlen_q=max_seqlen,
            max_seqlen_k=max_seqlen_kv,
            dropout_p=self.dropout,
            causal=self.causal,
        ).type_as(x)
        out = rearrange(out, "t h c -> t (h c)")
        x = x + self.to_out(out)
        x = x + self.ffn(x)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, depth: int, dim: int, dim_context: int = None, head_dim=64, dropout=0.0, expansion=4.0, causal=False):
        super().__init__()
        transformer = partial(
            TransformerLayer,
            dim=dim,
            dim_context=dim_context,
            head_dim=head_dim,
            dropout=dropout,
            expansion=expansion,
            causal=causal,
        )

        self.layers = nn.ModuleList()
        for _ in range(depth):
            self.layers.append(transformer())

    def forward(
        self,
        x: Tensor,
        seqlen: Tensor,
        max_seqlen: int,
        context: Tensor = None,
        seqlen_kv: Tensor = None,
        max_seqlen_kv: int = None,
    ):
        for layer in self.layers:
            x = layer(x, seqlen, max_seqlen, context, seqlen_kv, max_seqlen_kv)
        return x


class TransformerBlockBatched(nn.Module):
    def __init__(self, depth: int, dim: int, dim_context: int = None, head_dim=64, dropout=0.0, expansion=4.0, causal=False):
        super().__init__()
        self.block = TransformerBlock(
            depth=depth,
            dim=dim,
            dim_context=dim_context,
            head_dim=head_dim,
            dropout=dropout,
            expansion=expansion,
            causal=causal,
        )

    def forward(self, x: Tensor, context: Tensor = None):
        """
        - input x: b n c
        - input context: b m dim_context
        """
        context = context if context is not None else x

        B, N, C = x.shape
        x = rearrange(x, "b n c -> (b n) c")
        seqlen = th.linspace(0, B * N, B + 1, dtype=th.int32, device=x.device)
        max_seqlen = N

        B, M, C = context.shape
        context = rearrange(context, "b m c -> (b m) c")
        seqlen_kv = th.linspace(0, B * M, B + 1, dtype=th.int32, device=context.device)
        max_seqlen_kv = M

        x = self.block(x, seqlen, max_seqlen, context, seqlen_kv, max_seqlen_kv)  # (b n) c
        x = rearrange(x, "(b n) c -> b n c", b=B)
        return x
