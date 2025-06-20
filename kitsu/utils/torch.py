import gc

import numpy as np
import torch as th
from torch import Tensor, FloatTensor


def get_device():
    if th.cuda.is_available():
        return th.device("cuda")
    elif th.backends.mps.is_available():
        return th.device("mps")
    elif "xla" in th.device.__dict__:
        return th.device("xla")
    else:
        return th.device("cpu")


device = get_device()


def all_to(data, device):
    if isinstance(data, Tensor):
        return data.to(device)
    elif isinstance(data, np.ndarray):
        return th.from_numpy(data).to(device)
    elif isinstance(data, dict):
        return {k: all_to(v, device) for k, v in data.items()}
    elif isinstance(data, (list, tuple)):
        return type(data)(all_to(x, device) for x in data)
    elif isinstance(data, set):
        return {all_to(x, device) for x in data}
    else:
        return data


def clear_memory():
    """
    Clears cached GPU memory based on the available PyTorch backend (CUDA or MPS).
    If no GPU is available, it prints a message.
    """
    if th.cuda.is_available():
        th.cuda.empty_cache()
    elif th.backends.mps.is_available():
        th.mps.empty_cache()

    gc.collect()


def get_mixed_precision_dtype(device):
    # Check for bfloat16 support on CUDA device
    if device.type == "cuda":
        major, minor = th.cuda.get_device_capability(device)
        # Ampere (8.0+) and newer support bfloat16
        if major >= 8:
            return th.bfloat16
        else:
            return th.float16
    else:
        # On CPU, bfloat16 is supported but autocast for CPU is less common
        return th.float16


class PadRight:
    def __init__(self, x, attn_mask):
        self.original_mask = attn_mask
        self.x_shape = x.shape
        self.indices1 = attn_mask.nonzero(as_tuple=True)[0]
        self.indices2 = sum([list(range(n)) for n in attn_mask.sum(1).tolist()], [])

        self.new_attn_mask = th.zeros_like(attn_mask)
        self.new_attn_mask[self.indices1, self.indices2] = True

    def pad(self, x: FloatTensor):
        x_flat = x.contiguous()
        y = th.zeros_like(x_flat)
        y[self.indices1, self.indices2] = x_flat[self.original_mask]
        return y

    def unpad(self, y: FloatTensor):
        y_flat = y.contiguous()
        x = y.new_zeros(self.x_shape)
        x[self.original_mask] = y_flat[self.new_attn_mask]
        return x
