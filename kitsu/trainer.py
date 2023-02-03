import math
import random
import sys
from abc import ABCMeta, abstractmethod
from collections import OrderedDict
from contextlib import contextmanager
from copy import deepcopy
from functools import reduce
from typing import List, Sequence, Tuple

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.cuda.amp.autocast_mode import autocast
from torch.cuda.amp.grad_scaler import GradScaler
from torch.nn.parallel.distributed import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from kitsu import utils
from kitsu.logger import CustomLogger
from kitsu.utils.data import infinite_dataloader
from kitsu.utils.ema import ema
from kitsu.utils.optim import ESAM, SAM


class BasePreprocessor(metaclass=ABCMeta):
    def __init__(self, device) -> None:
        self.device = device

    @staticmethod
    def _to(x, device):
        if isinstance(x, torch.Tensor):
            x = x.to(device, non_blocking=True)
        elif isinstance(x, np.ndarray):
            x = torch.from_numpy(x).to(device, non_blocking=True)
        elif isinstance(x, List):
            x = [BasePreprocessor._to(item, device) for item in x]
        return x

    def batch_to_device(self, batch):
        out = []
        if isinstance(batch, list):
            for x in batch:
                x = self._to(x, self.device)
                out.append(x)
            return out
        else:
            x = self._to(batch, self.device)
            return x

    @abstractmethod
    def __call__(self, batch, augmentation=False):
        pass


class BaseWorker(metaclass=ABCMeta):
    def __init__(self, args) -> None:
        self.args = args

    @property
    def rank(self):
        return self.args.rank

    @property
    def rankzero(self):
        return self.args.rank == 0

    @property
    def world_size(self):
        return self.args.world_size

    @property
    def ddp(self):
        return self.args.ddp

    @property
    def log(self) -> CustomLogger:
        return self.args.log

    def collect_log(self, s, prefix="", postfix=""):
        keys = list(s.log.keys())
        if self.ddp:
            g = s.log.loss.new_tensor([self._t2f(s.log[k]) for k in keys], dtype=torch.float) * s.n
            dist.all_reduce(g)
            n = s.n * self.args.world_size
            g /= n

            out = OrderedDict()
            for k, v in zip(keys, g.tolist()):
                out[prefix + k + postfix] = v
        else:
            out = OrderedDict()
            for k in keys:
                out[prefix + k + postfix] = self._t2f(s.log[k])
            n = s.n
        return n, out

    def g_to_msg(self, g):
        msg = ""
        for k, v in g.items():
            msg += " %s:%.4f" % (k, v)
        return msg[1:]

    def _t2f(self, x):
        if isinstance(x, torch.Tensor):
            return x.item()
        else:
            return x


class BaseTrainer(BaseWorker):
    def __init__(
        self,
        args,
        n_samples_per_class: int = 10,
        find_unused_parameters: bool = False,
        sample_at_least_per_epochs: int = None,  # sampling is done not so frequently
        mixed_precision: bool = False,
        clip_grad: float = 0.0,
        num_saves: int = 5,  # save only latest n checkpoints
        epochs_to_save: int = 0,  # save checkpoint and do sampling after n epochs
        use_sync_bn: bool = False,
        monitor: str = "loss",
        small_is_better: bool = True,
        use_sam: bool = False,  # Sharpness-Aware Minimization
        use_esam: bool = False,  # Efficient Sharpness-aware Minimization
        save_only_improved: bool = True,
    ) -> None:
        assert not (mixed_precision and (use_sam or use_esam))
        # assert not (use_sam and use_esam)

        super().__init__(args)

        self.n_samples_per_class = n_samples_per_class
        self.find_unused_parameters = find_unused_parameters
        self.sample_at_least_per_epochs = sample_at_least_per_epochs
        self.mixed_precision = mixed_precision
        self.clip_grad = clip_grad
        self.num_saves = num_saves
        self.epochs_to_save = epochs_to_save
        self.use_sync_bn = use_sync_bn
        self.monitor = monitor
        self.small_is_better = small_is_better
        self.use_sam = use_sam
        self.use_esam = use_esam
        self.save_only_improved = save_only_improved

        if self.mixed_precision:
            self.scaler = GradScaler()

        self.best = math.inf if self.small_is_better else -math.inf
        self.best_epoch = -1

        self.build_network()
        self.build_dataset()
        self.build_sample_idx()
        self.build_preprocessor()

        if self.args.debug:
            self.args.epochs = 2
            self.epochs_to_save = 0

    @property
    def model(self):
        return self.model_src

    def build_network(self):
        self.model_src = utils.instantiate_from_config(self.args.model).cuda()
        if self.ddp:
            if self.use_sync_bn:
                self.model_src = nn.SyncBatchNorm.convert_sync_batchnorm(self.model_src)
            self.model_optim = DDP(
                self.model_src,
                device_ids=[self.args.gpu],
                find_unused_parameters=self.find_unused_parameters,
            ).cuda()
        else:
            self.model_optim = self.model_src

        self.optim = utils.instantiate_from_config(self.args.optim, self.model_optim.parameters())

        if self.use_sam:
            self.optim = SAM(self.model_optim.parameters(), self.optim)
        elif self.use_esam:
            self.optim = ESAM(self.model_optim.parameters(), self.optim)

        if "sched" in self.args:
            self.sched = utils.instantiate_from_config(self.args.sched, self.optim)
        else:
            self.sched = None

        # self.log.info(self.model)
        self.log.info("Model Params: %.2fM" % (self.model_params / 1e6))

    def build_dataset(self):
        dls: Sequence[Dataset] = utils.instantiate_from_config(self.args.dataset)
        if len(dls) == 3:
            self.dl_train, self.dl_valid, self.dl_test = dls
            l1, l2, l3 = len(self.dl_train.dataset), len(self.dl_valid.dataset), len(self.dl_test.dataset)
            self.log.info("Load %d train, %d valid, %d test items" % (l1, l2, l3))
        elif len(dls) == 2:
            self.dl_train, self.dl_valid = dls
            l1, l2 = len(self.dl_train.dataset), len(self.dl_valid.dataset)
            self.log.info("Load %d train, %d valid items" % (l1, l2))
        else:
            raise NotImplementedError

    def build_preprocessor(self):
        self.preprocessor: BasePreprocessor = utils.instantiate_from_config(self.args.preprocessor, device=self.device)

    def build_sample_idx(self):
        pass

    def save(self, out_path):
        data = {
            "optim": self.optim.state_dict(),
            "model": self.model_src.state_dict(),
        }
        torch.save(data, str(out_path))

    def step(self, s):
        pass

    @property
    def device(self):
        return next(self.model_src.parameters()).device

    @property
    def model_params(self):
        model_size = 0
        for param in self.model_src.parameters():
            if param.requires_grad:
                model_size += param.data.nelement()
        return model_size

    def on_train_batch_end(self, s):
        pass

    def on_valid_batch_end(self, s):
        pass

    def train_epoch(self, dl: "DataLoader", prefix="Train"):
        self.model_optim.train()
        o = utils.AverageMeters()

        if self.rankzero:
            desc = f"{prefix} [{self.epoch:04d}/{self.args.epochs:04d}]"
            t = tqdm(total=len(dl.dataset), ncols=150, file=sys.stdout, desc=desc, leave=True)
        for batch in dl:
            s = self.preprocessor(batch, augmentation=True)
            with autocast(self.mixed_precision):
                self.step(s)

            if self.mixed_precision:
                self.scaler.scale(s.log.loss).backward()
                if self.clip_grad > 0:  # gradient clipping
                    self.scaler.unscale_(self.optim)
                    nn.utils.clip_grad.clip_grad_norm_(self.model_optim.parameters(), self.clip_grad)
                self.scaler.step(self.optim)
                self.scaler.update()
            else:
                s.log.loss.backward()
                if self.clip_grad > 0:  # gradient clipping
                    nn.utils.clip_grad.clip_grad_norm_(self.model_optim.parameters(), self.clip_grad)

                if self.use_sam or self.use_esam:
                    self.optim.first_step(zero_grad=True)
                    s = self.preprocessor(batch, augmentation=True)
                    with autocast(self.mixed_precision):
                        self.step(s)
                    s.log.loss.backward()
                    self.optim.second_step(zero_grad=False)
                else:
                    self.optim.step()
            self.optim.zero_grad()

            self.step_sched(is_on_batch=True)

            n, g = self.collect_log(s)
            o.update_dict(n, g)
            if self.rankzero:
                t.set_postfix_str(o.to_msg(), refresh=False)
                t.update(min(n, t.total - t.n))

            self.on_train_batch_end(s)

            if self.args.debug:
                break
        if self.rankzero:
            t.close()
        return o

    @torch.no_grad()
    def valid_epoch(self, dl: "DataLoader", prefix="Valid"):
        self.model_optim.eval()
        o = utils.AverageMeters()

        if self.rankzero:
            desc = f"{prefix} [{self.epoch:04d}/{self.args.epochs:04d}]"
            t = tqdm(total=len(dl.dataset), ncols=150, file=sys.stdout, desc=desc, leave=True)
        for batch in dl:
            s = self.preprocessor(batch, augmentation=False)
            self.step(s)

            n, g = self.collect_log(s)
            o.update_dict(n, g)
            if self.rankzero:
                t.set_postfix_str(o.to_msg(), refresh=False)
                t.update(min(n, t.total - t.n))

            self.on_valid_batch_end(s)

            if self.args.debug:
                break
        if self.rankzero:
            t.close()
        return o

    @torch.no_grad()
    def evaluation(self, *o_lst):
        assert self.monitor in o2.data, f"No monitor {self.monitor} in validation results: {list(o2.data.keys())}"

        self.step_sched(o_lst[0][self.monitor], is_on_epoch=True)

        improved = False
        if self.rankzero:  # scores are not calculated in other nodes
            flag = ""
            if (
                (self.small_is_better and o_lst[0][self.monitor] < self.best)
                or (not self.small_is_better and o_lst[0][self.monitor] > self.best)
                or (
                    self.sample_at_least_per_epochs is not None
                    and (self.epoch - self.best_epoch) >= self.sample_at_least_per_epochs
                )
            ):
                if self.small_is_better:
                    if self.best > o_lst[0][self.monitor]:
                        self.best = o_lst[0][self.monitor]
                        improved = True
                else:
                    if self.best < o_lst[0][self.monitor]:
                        self.best = max(self.best, o_lst[0][self.monitor])
                        improved = True

                self.best_epoch = self.epoch
                self.save(self.args.exp_path / "best_ep{:04d}.pth".format(self.epoch))
                saved_files = sorted(list(self.args.exp_path.glob("best_ep*.pth")))
                if len(saved_files) > self.num_saves:
                    to_deletes = saved_files[: len(saved_files) - self.num_saves]
                    for to_delete in to_deletes:
                        utils.io.try_remove_file(str(to_delete))

                flag = "*"
                improved = self.epoch > self.epochs_to_save or self.args.debug or not self.save_only_improved

            msg = "Epoch[%03d/%03d]" % (self.epoch, self.args.epochs)
            msg += f" {self.monitor}[" + ";".join([o._get(self.monitor) for o in o_lst]) + "]"
            msg += " (best:%.4f%s)" % (self.best, flag)

            keys = reduce(lambda x, o: x | set(o.data.keys()), o_lst, set())
            keys = sorted(list(filter(lambda x: x != self.monitor, keys)))

            for k in keys:
                msg += f" {k}[" + ";".join([o._get(k) for o in o_lst]) + "]"

            self.log.info(msg)
            self.log.flush()

        # share improved condition with other nodes
        if self.ddp:
            improved = torch.tensor([improved], device="cuda")
            dist.broadcast(improved, 0)

        return improved

    def fit_loop(self):
        o1 = self.train_epoch(self.dl_train)
        o2 = self.valid_epoch(self.dl_valid)
        improved = self.evaluation(o2, o1)
        if improved:
            self.sample()

    def fit(self):
        for self.epoch in range(1, self.args.epochs + 1):
            self.fit_loop()

    def sample(self):
        pass

    def step_sched(self, metric=None, is_on_batch=False, is_on_epoch=False):
        if self.sched is None:
            return
        if (is_on_batch and self.args.sched.step_on_batch) or (is_on_epoch and self.args.sched.step_on_epoch):
            if self.sched.__class__.__name__ in ("ReduceLROnPlateau", "ReduceLROnPlateauWithWarmup"):
                assert metric is not None
                self.sched.step(metric)
            else:
                self.sched.step()


class BaseTrainerEMA(BaseTrainer):
    def __init__(self, *args, ema_decay: float, **kwargs):
        super().__init__(*args, **kwargs)
        self.ema_decay = ema_decay

        self._ema_state = False

    @contextmanager
    def ema_state(self, activate=True):
        previous = self._ema_state
        self._ema_state = activate
        yield
        self._ema_state = previous

    def build_network(self):
        super().build_network()
        self.model_ema = deepcopy(self.model_src)
        self.model_ema.load_state_dict(self.model_src.state_dict())
        self.model_ema.eval().requires_grad_(False)

    def on_train_batch_end(self, s):
        super().on_train_batch_end(s)
        ema(self.model_src, self.model_ema, self.ema_decay)

    def save(self, out_path):
        data = {
            "optim": self.optim.state_dict(),
            "model": self.model_src.state_dict(),
            "model_ema": self.model_ema.state_dict(),
        }
        torch.save(data, str(out_path))


class StepTrainer(BaseTrainer):
    def __init__(
        self,
        args,
        valid_per_steps,
        **kwargs,
    ) -> None:
        super().__init__(args, **kwargs)

        self.valid_per_steps = valid_per_steps

    def train_batch(self, batch, o: utils.AverageMeters):
        s = self.preprocessor(batch, augmentation=True)
        with autocast(self.mixed_precision):
            self.step(s)

        if self.mixed_precision:
            self.scaler.scale(s.log.loss).backward()
            if self.clip_grad > 0:  # gradient clipping
                self.scaler.unscale_(self.optim)
                nn.utils.clip_grad.clip_grad_norm_(self.model_optim.parameters(), self.clip_grad)
            self.scaler.step(self.optim)
            self.scaler.update()
        else:
            s.log.loss.backward()
            if self.clip_grad > 0:  # gradient clipping
                nn.utils.clip_grad.clip_grad_norm_(self.model_optim.parameters(), self.clip_grad)
            self.optim.step()
        self.optim.zero_grad()

        n, g = self.collect_log(s)
        o.update_dict(n, g)

        self.on_train_batch_end(s)
        self.step_sched(is_on_batch=True)

    @torch.no_grad()
    def valid_epoch(self, dl: "DataLoader", prefix="Valid"):
        o = utils.AverageMeters()
        desc = f"{prefix} [{self.epoch:04d}/{self.args.epochs:04d}]"

        with tqdm(total=len(dl.dataset), ncols=150, file=sys.stdout, desc=desc, leave=False, disable=not self.rankzero) as pbar:
            for batch in dl:
                s = self.preprocessor(batch, augmentation=False)
                self.step(s)

                n, g = self.collect_log(s)
                o.update_dict(n, g)

                pbar.set_postfix_str(o.to_msg(), refresh=False)
                pbar.update(min(n, pbar.total - pbar.n))

                self.on_valid_batch_end(s)

                if self.args.debug:
                    break
        return o

    @torch.no_grad()
    def evaluation(self, *o_lst):
        self.step_sched(o_lst[0][self.monitor], is_on_epoch=True)

        improved = False
        if self.rankzero:  # scores are not calculated in other nodes
            flag = ""
            if (
                (self.small_is_better and o_lst[0][self.monitor] < self.best)
                or (not self.small_is_better and o_lst[0][self.monitor] > self.best)
                or (
                    self.sample_at_least_per_epochs is not None
                    and (self.epoch - self.best_epoch) >= self.sample_at_least_per_epochs
                )
            ):
                if self.small_is_better:
                    if self.best > o_lst[0][self.monitor]:
                        self.best = o_lst[0][self.monitor]
                        improved = True
                else:
                    if self.best < o_lst[0][self.monitor]:
                        self.best = max(self.best, o_lst[0][self.monitor])
                        improved = True

                self.best_epoch = self.epoch
                self.save(self.args.exp_path / "best_ep{:06d}.pth".format(self.epoch))
                saved_files = sorted(list(self.args.exp_path.glob("best_ep*.pth")))
                if len(saved_files) > self.num_saves:
                    to_deletes = saved_files[: len(saved_files) - self.num_saves]
                    for to_delete in to_deletes:
                        utils.io.try_remove_file(str(to_delete))

                flag = "*"
                improved = self.epoch > self.epochs_to_save or self.args.debug or not self.save_only_improved

            msg = f"Epoch[%03d/%03d]" % (self.epoch, self.args.epochs)
            msg += f" {self.monitor}[" + ";".join([o._get(self.monitor) for o in o_lst]) + "]"
            msg += " (best:%.4f%s)" % (self.best, flag)

            keys = reduce(lambda x, o: x | set(o.data.keys()), o_lst, set())
            keys = sorted(list(filter(lambda x: x != self.monitor, keys)))

            for k in keys:
                msg += f" {k}[" + ";".join([o._get(k) for o in o_lst]) + "]"

            self.log.info(msg)
            self.log.flush()

        # share improved condition with other nodes
        if self.ddp:
            improved = torch.tensor([improved], device="cuda")
            dist.broadcast(improved, 0)

        return improved

    @property
    def _is_eval_stage(self):
        return self.valid_per_steps is not None and (self.epoch % self.valid_per_steps == 0 or self.args.debug)

    @torch.no_grad()
    def stage_eval(self, o_train):
        o_valid = self.valid_epoch(self.dl_valid)
        improved = self.evaluation(o_valid, o_train)

        if improved:
            self.sample()

    def fit(self):
        o_train = utils.AverageMeters()
        with tqdm(total=self.args.epochs, ncols=150, file=sys.stdout, disable=not self.rankzero, desc="Step") as pbar:
            self.model_optim.train()
            for self.epoch, batch in enumerate(infinite_dataloader(self.dl_train), 1):
                self.train_batch(batch, o_train)
                pbar.set_postfix_str(o_train.to_msg())

                if self._is_eval_stage:
                    print()
                    self.model_optim.eval()
                    self.stage_eval(o_train)
                    self.model_optim.train()
                    o_train = utils.AverageMeters()

                pbar.update()

                if self.args.debug and self.epoch >= 2:
                    break
                if self.epoch >= self.args.epochs:
                    break


class StepTrainerEMA(StepTrainer):
    def __init__(self, *args, ema_decay: float, **kwargs):
        super().__init__(*args, **kwargs)
        self.ema_decay = ema_decay

        self._ema_state = False

    @contextmanager
    def ema_state(self, activate=True):
        previous = self._ema_state
        self._ema_state = activate
        yield
        self._ema_state = previous

    def build_network(self):
        super().build_network()

        self.model_ema = deepcopy(self.model_src)
        self.model_ema.load_state_dict(self.model_src.state_dict())
        self.model_ema.requires_grad_(False)

    def on_train_batch_end(self, s):
        super().on_train_batch_end(s)
        ema(self.model_src, self.model_ema, self.ema_decay)

    def save(self, out_path):
        data = {
            "optim": self.optim.state_dict(),
            "model": self.model_src.state_dict(),
            "model_ema": self.model_ema.state_dict(),
        }
        torch.save(data, str(out_path))

    @torch.no_grad()
    def stage_eval(self, o_train):
        o_valid = self.valid_epoch(self.dl_valid)
        with self.ema_state():
            o_valid_ema = self.valid_epoch(self.dl_valid)
        improved = self.evaluation(o_valid, o_valid_ema, o_train)

        if improved:
            self.stage_save()
            self.sample()
