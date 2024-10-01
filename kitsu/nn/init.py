import torch as th
import torch.nn as nn
import torch.nn.init as init
from timm.models.layers import trunc_normal_

_module_list = (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.Linear, nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d)


def dm(module: nn.Module, last=False):
    gain = 1e-5 if last else 1.0

    for m in module.modules():
        if isinstance(m, _module_list):
            init.xavier_normal_(m.weight, gain=gain)
            if m.bias is not None:
                init.zeros_(m.bias)


def convnextv2(module: nn.Module, last=False):
    for m in module.modules():
        if isinstance(m, _module_list):
            trunc_normal_(m, std=0.02)
            if m.bias is not None:
                init.zeros_(m.bias)
