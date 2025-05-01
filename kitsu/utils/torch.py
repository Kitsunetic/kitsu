import torch as th


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
