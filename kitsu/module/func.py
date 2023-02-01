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
