import torch as th
import torch.nn.functional as F


def focal_loss(inputs, targets, alpha=1, gamma=2, reduction="mean"):
    ce_loss = F.cross_entropy(inputs, targets, reduction="none")
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
