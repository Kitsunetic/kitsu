import importlib
import os
import random
from collections import defaultdict
from copy import deepcopy
from typing import Callable, Sequence, Union

import numpy as np
import torch
import torch as th
import torch.nn as nn
from easydict import EasyDict
from torch import Tensor
from torchvision.utils import make_grid

__all__ = [
    "AverageMeter",
    "AverageMeters",
    "seed_everything",
    "find_free_port",
    "get_model_params",
    "instantiate_from_config",
    "get_obj_from_str",
    "BlackHole",
    "tensor_to_image",
    "safe_to_tensor",
    "cummul",
    "DefaultEasyDict",
    "partial_loose",
]


class AverageMeter(object):
    def __init__(self):
        self.sum = 0
        self.cnt = 0
        self.avg = 0

    def update(self, val, n=1):
        if n > 0:
            self.sum += val * n
            self.cnt += n
            self.avg = self.sum / self.cnt

    def get(self):
        return self.avg

    def __call__(self):
        return self.avg


class AverageMeters:
    def __init__(self, *keys) -> None:
        # self.data = OrderedDict({key: AverageMeter() for key in keys})
        self.data = defaultdict(AverageMeter)
        for k in keys:
            self.data[k]

    def __getitem__(self, key):
        return self.data[key]()

    def __getattr__(self, key):
        return self.data[key]()

    def update_dict(self, n, g):
        for k, v in g.items():
            self.data[k].update(v, n)

    def _get(self, k):
        if k in self.data:
            return f"{self.data[k]():.4f}"
        else:
            return "_"

    def to_msg(self, format="%s:%.4f"):
        msgs = []
        for k, v in self.data.items():
            if k == "loss":
                msgs = [format % (k, v())] + msgs
            else:
                msgs.append(format % (k, v()))
        return " ".join(msgs)


def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False


def find_free_port():
    import socket

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # Binding to port 0 will cause the OS to find an available port for us
    sock.bind(("", 0))
    port = sock.getsockname()[1]
    sock.close()
    # NOTE: there is still a chance the port could be taken by other processes.
    return port


def get_model_params(model):
    model_size = 0
    for param in model.parameters():
        model_size += param.data.nelement()
    return model_size


def _parse_pyinstance_dict(params: dict):
    out_dict = EasyDict()

    for p, v in params.items():
        if p == "__pyinstance__":
            inst = instantiate_from_config(v)
            return inst
        elif isinstance(v, dict):
            out_dict[p] = _parse_pyinstance_dict(v)
        elif isinstance(v, (list, tuple)):
            out_dict[p] = _parse_pyinstance_list(v)
        else:
            out_dict[p] = v

    return out_dict


def _parse_pyinstance_list(params: list):
    out_list = []

    for v in params:
        if isinstance(v, dict):
            out_list.append(_parse_pyinstance_dict(v))
        elif isinstance(v, (list, tuple)):
            out_list.append(_parse_pyinstance_list(v))
        else:
            out_list.append(v)

    return out_list


def instantiate_from_config(config: dict, *args, **kwargs):
    config = deepcopy(config)

    # https://github.com/CompVis/latent-diffusion/blob/a506df5756472e2ebaf9078affdde2c4f1502cd4/ldm/util.py#L78
    if not "target" in config:
        raise KeyError("Expected key `target` to instantiate.")

    # parse __pyinstance__
    argums = config.get("argums", list())
    argums = _parse_pyinstance_list(argums)
    params = config.get("params", dict())
    params = _parse_pyinstance_dict(params)

    return get_obj_from_str(config["target"])(*argums, *args, **params, **kwargs)


def get_obj_from_str(string, reload=False):
    # https://github.com/CompVis/latent-diffusion/blob/a506df5756472e2ebaf9078affdde2c4f1502cd4/ldm/util.py#L88
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


def tensor_to_image(images, nrow):
    # images: b 3 h w, [-1, 1]
    grid = make_grid(images, nrow=nrow).permute(1, 2, 0)  # H W 3 [-1, 1]
    # (x+1)/2 * 255 + 0.5 = 127.5x + 128, (반올림이 되게 하기 위해 0.5를 더함, 안 더하면 내림이 됨)
    grid = grid.mul_(127.5).add_(128).clamp_(0, 255).to("cpu", torch.uint8).numpy()
    return grid


class BlackHole(int):
    def __setattr__(self, *args, **kwargs):
        pass

    def __call__(self, *args, **kwargs):
        return self

    def __getattr__(self, *args, **kwargs):
        return self

    def __enter__(self, *args, **kwargs):
        return self

    def __exit__(self, *args, **kwargs):
        return self

    def __getitem__(self, *args, **kwargs):
        return self


def tensor_to_image(x: Tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """
    ### input
    - x: (b 3 h w) or (3 h w)
    """
    xdim = x.dim()
    if xdim == 3:
        x = x[None]

    if not isinstance(mean, Tensor):
        mean = x.new_tensor(mean).view(1, 3, 1, 1)
    else:
        mean = mean.to(x).view(1, 3, 1, 1)
    if not isinstance(std, Tensor):
        std = x.new_tensor(std).view(1, 3, 1, 1)
    else:
        std = std.to(x).view(1, 3, 1, 1)
    x = x * std + mean
    x = x.detach().mul(255).add_(0.5).clamp_(0, 255).type(th.uint8).permute(0, 2, 3, 1).cpu().numpy()

    if xdim == 3:
        x = x[0]

    return x


def safe_to_tensor(x, device="cpu"):
    non_blocking = device != "cpu"

    if isinstance(x, np.ndarray):
        return th.from_numpy(x).to(device, non_blocking=non_blocking)
    elif isinstance(x, th.Tensor):
        return x.to(device, non_blocking=non_blocking)
    elif isinstance(x, (list, tuple)):
        return th.tensor(x, device=device)
    elif isinstance(x, dict):
        return {k: safe_to_tensor(v, device=device) for k, v in x.items()}
    return x


def cummul(*x):
    y = 1
    for v in x:
        y *= v
    return y


def get_model_params(model: nn.Module):
    model_size = 0
    for param in model.parameters():
        if param.requires_grad:
            model_size += param.data.nelement()
    return model_size


class DefaultEasyDict(defaultdict):
    """DefulatDict + EasyDict"""

    def __init__(self, *args: Sequence[Union[Callable, dict]], **kwargs):
        default_factory = None

        for arg in args:
            if isinstance(arg, dict):
                for k, v in arg.items():
                    self.__setattr__(k, v)
            elif isinstance(arg, Callable):
                if default_factory is not None:
                    raise NotImplementedError(f"Only a default factory can be given.")
                default_factory = arg
            else:
                raise NotImplementedError(f"Unknown input data type: {arg}.")

        for k, v in kwargs.items():
            self.__setattr__(k, v)

        if default_factory is not None:
            super().__init__(default_factory)

    def __setattr__(self, name, value):
        if isinstance(value, (list, tuple)):
            value = type(value)(self.__class__(x) if isinstance(x, dict) else x for x in value)
        elif isinstance(value, dict) and not isinstance(value, DefaultEasyDict):
            value = EasyDict(value)

        self.__setitem__(name, value)

    def __setitem__(self, name, value):
        if name == "_ipython_canary_method_should_not_exist_":
            return
        super().__setitem__(name, value)

    def __getattr__(self, name):
        return self.__getitem__(name)

    def update(self, e=None, **f):
        d = e or dict()
        d.update(f)
        for k in d:
            setattr(self, k, d[k])

    def pop(self, k, *args):
        if hasattr(self, k):
            delattr(self, k)
        return super(EasyDict, self).pop(k, *args)

    def as_dict(self):
        out = {}
        for k, v in self.items():
            out[k] = self._as_dict(v)
        return out

    def _as_dict(self, v):
        if isinstance(v, list):
            return [self._as_dict(x) for x in v]
        elif isinstance(v, tuple):
            return (self._as_dict(x) for x in v)
        elif isinstance(v, set):
            return {self._as_dict(x) for x in v}
        elif isinstance(v, DefaultEasyDict):
            return v._as_dict()
        else:
            return v


def partial_loose(fn: Callable, **kwargs):
    def wrapper(*args, **kwargs_):
        kwargs_updated = kwargs.copy()
        kwargs_updated.update(kwargs_)
        return fn(*args, **kwargs_updated)

    return wrapper
