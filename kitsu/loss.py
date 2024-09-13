import torch as th
import torch.nn.functional as F
from torch import Tensor


def focal_loss(inputs: Tensor, targets: Tensor, alpha=1, gamma=2, reduction="mean", label_smoothing=0.0):
    """
    Args:
        inputs (Tensor): b c
        targets (Tensor): b, int64
    """
    if label_smoothing <= 0.0:
        ce_loss = F.cross_entropy(inputs, targets, reduction="none")
        p_t = ce_loss.neg().exp()
        focal_loss = alpha * (1 - p_t) ** gamma * ce_loss
    else:
        C = inputs.size(-1)
        if label_smoothing > 0.0:
            targets = (1 - label_smoothing) * F.one_hot(targets, C) + label_smoothing / C

        log_probs = th.log_softmax(inputs, -1)
        ce_loss = -(targets * log_probs).sum(-1)  # b
        p_t = ce_loss.neg().exp()
        focal_loss = alpha * (1 - p_t) ** gamma * ce_loss

    if reduction == "mean":
        return focal_loss.mean()
    elif reduction == "sum":
        return focal_loss.sum()
    else:
        return focal_loss


if __name__ == "__main__":
    x = th.randn(10000, 1000, device="cuda")
    y = th.randint(0, 1000, (10000,), device="cuda")
    print(focal_loss(x, y))
