import torch as th
import torch.nn as nn


def count_parameters(model: nn.Module) -> int:
    """
    Count the total number of parameters in a given nn.Module.

    Args:
        model (nn.Module): The PyTorch model to analyze.

    Returns:
        int: The total number of parameters.
    """
    return sum(p.numel() for p in model.parameters())
