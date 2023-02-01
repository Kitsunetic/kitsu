from collections import OrderedDict
from typing import Union

import torch.nn as nn


def ema(source: Union[OrderedDict, nn.Module], target: Union[OrderedDict, nn.Module], decay: float):
    if isinstance(source, nn.Module):
        source = source.state_dict()
    if isinstance(target, nn.Module):
        target = target.state_dict()
    for key in source.keys():
        target[key].data.copy_(target[key].data * decay + source[key].data * (1 - decay))
