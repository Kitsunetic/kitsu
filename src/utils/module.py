from functools import partial

import torch.nn as nn
import torch.nn.init as init


class Partial(nn.Module):
    def __init__(self, __func, *args, **kwargs) -> None:
        super().__init__()
        self.fn = partial(__func, *args, **kwargs)

    def forward(self, *args, **kwargs):
        return self.fn(*args, **kwargs)


class Lambda(nn.Module):
    def __init__(self, __func) -> None:
        super().__init__()
        self.__func = __func

    def forward(self, *args, **kwargs):
        return self.__func(*args, **kwargs)


def _zero_init_(m: nn.Module):
    if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.Linear)):
        init.zeros_(m.weight)
        if hasattr(m, "bias") and m.bias is not None:
            init.zeros_(m.bias)


def zero_init(m: nn.Module):
    if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.Linear)):
        _zero_init_(m)
    else:
        m.apply(_zero_init_)


def _loose_zero_init_(m: nn.Module):
    if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.Linear)):
        init.xavier_uniform_(m.weight, gain=1e-5)
        if hasattr(m, "bias") and m.bias is not None:
            init.zeros_(m.bias)


def loose_zero_init(m: nn.Module):
    if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.Linear)):
        _loose_zero_init_(m)
    else:
        m.apply(_loose_zero_init_)
