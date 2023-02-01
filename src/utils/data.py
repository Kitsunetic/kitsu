from typing import Sequence

import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from torchvision.datasets import CIFAR100
from torchvision.transforms import ToTensor


def first(xs):
    for i, x in enumerate(xs):
        yield x, i == 0


def build_dataloaders(batch_size: int, num_workers: int, *dss: Sequence[Dataset], **kwargs) -> Sequence[DataLoader]:
    dl_kwargs = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=True, persistent_workers=num_workers > 0)
    dl_kwargs.update(kwargs)
    if dist.is_initialized() and dist.get_world_size() > 1:
        samplers = [DistributedSampler(ds, shuffle=shuffle) for ds, shuffle in first(dss)]
        dls = [DataLoader(ds, sampler=sampler, **dl_kwargs) for ds, sampler in zip(dss, samplers)]
    else:
        dls = [DataLoader(ds, shuffle=shuffle, **dl_kwargs) for ds, shuffle in first(dss)]

    if len(dls) == 1:
        return dls[0]
    else:
        return dls
