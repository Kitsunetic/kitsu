import numpy as np
import matplotlib.pyplot as plt

import torch as th


__all__ = ["plot", "normalize", "standardize"]


def _safe_to_numpy(x):
    if isinstance(x, th.Tensor):
        x = x.detach().cpu().numpy()
    elif not isinstance(x, np.ndarray):
        x = np.array(x)
    return x


def plot(*data, figsize=(8, 3), color=None, legend=None, save_path=None, no_show=False):
    """
    - x: 1D array in a type of numpy array or pytorch tensor.
    """
    if color is not None:
        assert len(color) == max(1, len(data) - 1)

    data = [_safe_to_numpy(x) for x in data]

    plt.figure(figsize=figsize)

    if len(data) == 1:
        if color is not None:
            plt.plot(data[0], color=color)
        else:
            plt.plot(data[0])
    else:
        x = data[0]
        for i in range(len(data) - 1):
            if color is not None:
                plt.plot(x, data[i + 1], color=color[i])
            else:
                plt.plot(x, data[i + 1])

    plt.tight_layout()

    if legend is not None:
        plt.legend(legend)

    if save_path is not None:
        plt.savefig(save_path)

    if not no_show:
        plt.show()


def normalize(x):
    x = _safe_to_numpy(x)
    amax, amin = x.max(), x.min()
    x = (x - amin) / (amax - amin)
    return x


def standardize(x):
    x = _safe_to_numpy(x)
    x = x - np.mean(x)
    x = x / np.std(x)
    return x
