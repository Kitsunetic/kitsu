import torch as th
import torch.nn as nn

from kitsu.nn.seqlen_utils import AttentionBatch, seqlen_to_batch_index

try:
    from torch_scatter import scatter_mean
except:
    pass


__all__ = [
    "PointLayerNorm",
    "PointInstanceNorm",
]


class PointLayerNorm(nn.Module):
    def __init__(self, dim, affine=True, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.affine = affine
        if affine:
            self.weight = nn.Parameter(th.ones(1, dim))
            self.bias = nn.Parameter(th.zeros(1, dim))

    def forward(self, x: AttentionBatch):
        B, N = len(x.seqlen) - 1, len(x.x)
        bidx = seqlen_to_batch_index(x.seqlen, x.max_seqlen)

        mean = scatter_mean(x.x.mean(1, keepdim=True), bidx, 0, dim_size=B)  # N c -> N 1 -> b 1
        x_hat = x.x - mean[bidx]  # N c, N 1 -> N c
        var = scatter_mean(x_hat**2, bidx, 0, dim_size=B).mean(-1, keepdim=True)  # N c -> b c -> b 1
        var = (var + self.eps).sqrt()[bidx]  # b 1 -> N 1

        x_hat = x_hat / var  # N c, N 1 -> N c
        if self.affine:
            x_hat = (x_hat * self.weight) + self.bias
        return x.new(x_hat)


class PointInstanceNorm(nn.Module):
    def __init__(self, dim, affine=False, eps=1e-5):
        super().__init__()
        self.affine = affine
        self.eps = eps
        if affine:
            self.weight = nn.Parameter(th.ones(1, dim))
            self.bias = nn.Parameter(th.zeros(1, dim))

    def forward(self, x: AttentionBatch):
        B = len(x.seqlen) - 1
        bidx = seqlen_to_batch_index(x.seqlen, x.max_seqlen)
        mean = scatter_mean(x.x, bidx, 0, dim_size=B)  # N c -> b c
        x_hat = x.x - mean[bidx]  # N c, N c -> N c
        var = scatter_mean(x_hat**2, bidx, 0, dim_size=B)  # b c
        var = (var + self.eps).sqrt()[bidx]  # N c

        x_hat = x_hat / var  # N c
        if self.affine:
            x_hat = (x_hat * self.weight) + self.bias
        return x.new(x_hat)
