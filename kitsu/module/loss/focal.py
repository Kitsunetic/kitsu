import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from kitsu.module.loss.utils import _reduce

__all__ = ["focal_loss", "FocalLoss"]


def focal_loss(gamma: float, logits: Tensor, targets: Tensor, reduction="mean") -> Tensor:
    """
    ### input:
    - logits: b c ...
    - targets: b ... (long)

    CE(p) = -\log(p)
    FL(p) = -(1 - p)^\gamma \log(p)
    """
    ce = F.cross_entropy(logits, targets, reduction="none")  # b ...
    scale = th.pow(1 - th.exp(-ce), gamma)
    loss = scale * ce
    return _reduce(loss, reduction)


class FocalLoss(nn.Module):
    def __init__(self, gamma: float) -> None:
        super().__init__()
        self.gamma = gamma

    def forward(self, logits: Tensor, targets: Tensor, reduction="mean") -> Tensor:
        return focal_loss(self.gamma, logits, targets, reduction=reduction)
