import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def _reduce(x: Tensor, reduction: str):
    if reduction == "mean":
        return x.mean()
    elif reduction == "none":
        return x
    elif reduction == "sum":
        return x.sum()
    elif reduction == "batchmean":
        return x.flatten(1).mean(1)
