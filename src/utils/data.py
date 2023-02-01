from math import inf
from typing import Sequence

import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset, DistributedSampler

__all__ = ["first", "build_dataloaders", "ChainDataset", "SubDataset", "infinite_dataloader"]


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
