from math import inf
from typing import Sequence, Union

import torch as th
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from torch.utils.data._utils.collate import default_collate

__all__ = ["first", "build_dataloaders", "ChainDataset", "SubDataset", "infinite_dataloader"]


def first(xs):
    for i, x in enumerate(xs):
        yield x, i == 0


def build_dataloaders(
    batch_size: int,
    num_workers: int,
    ds: Dataset,
    ddp=None,
    shuffle=False,
    pin_memory=None,
    persistent_workers=None,
    collate_fn=None,
    **kwargs,
) -> Union[DataLoader, Sequence[DataLoader]]:
    if pin_memory is None:
        pin_memory = th.cuda.is_available()
    if persistent_workers is None:
        persistent_workers = num_workers > 0
    dl_kwargs = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )
    dl_kwargs.update(kwargs)

    if ddp is None:
        ddp = dist.is_initialized() and dist.get_world_size() > 1

    if collate_fn is None:
        if hasattr(ds, "collate_fn"):
            collate_fn = ds.collate_fn
        else:
            collate_fn = default_collate

    if ddp:
        sampler = DistributedSampler(ds, shuffle=shuffle)
        dl = DataLoader(ds, sampler=sampler, collate_fn=collate_fn, **dl_kwargs)
    else:
        dl = DataLoader(ds, shuffle=shuffle, collate_fn=collate_fn, **dl_kwargs)

    return dl


class ChainDataset(Dataset):
    def __init__(self, *datasets) -> None:
        super().__init__()
        self.datasets = datasets
        self.lens = []
        self.cum_lens = []
        self.indices = []
        cum_n = 0
        for i, dataset in enumerate(self.datasets):
            n = len(dataset)
            self.lens.append(n)
            self.cum_lens.append(cum_n)
            self.indices += [i for _ in range(n)]
            cum_n += n
        self.total_len = sum(self.lens)

    def __len__(self):
        return self.total_len

    def __getitem__(self, idx):
        ds_idx = self.indices[idx]
        out = self.datasets[ds_idx][idx - self.cum_lens[ds_idx]]
        return out


class SubDataset(Dataset):
    def __init__(self, dataset, indices) -> None:
        super().__init__()
        self.dataset = dataset
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        subidx = self.indices[idx]
        return self.dataset[subidx]


def infinite_dataloader(dl, n_iters=inf):
    step = 0
    keep = True
    while keep:
        for batch in dl:
            yield batch
            step += 1
            if step > n_iters:
                keep = False
                break


def adjust_batch_size_and_num_workers(batch_size: int, num_workers: int, ddp: bool = None, world_size: int = None):
    if ddp is None:
        ddp = dist.is_initialized()
    if world_size is None:
        if ddp:
            world_size = dist.get_world_size()
        else:
            world_size = 1

    if ddp:
        batch_size = (batch_size + world_size - 1) // world_size
        num_workers = (num_workers + world_size - 1) // world_size

    return batch_size, num_workers
