import gc

import numpy as np
import torch as th
from torch import Tensor


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
