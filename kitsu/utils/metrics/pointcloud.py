import torch as th
from pytorch3d.ops import knn_points
from torch import Tensor


class PointDist:
    def __init__(self, a, b) -> None:
        """
        :params a: b n 3
        :params b: b n 3
        """
        self.a = a
        self.b = b
        self._knn_done = False

    def _calc_knn(self):
        self._dist1, _, self._knn1 = knn_points(self.a, self.b, return_nn=True, return_sorted=False)
        self._dist2, _, self._knn2 = knn_points(self.b, self.a, return_nn=True, return_sorted=False)
        # dists: b n 1
        # knn: b n 1 3

    def _reduce_result(self, x: Tensor, reduction="mean"):
        if reduction == "mean":
            return x.mean()
        elif reduction == "batchmean":
            return x.flatten(1).mean(1)
        else:
            raise NotImplementedError(reduction)

    def chamfer_l1(self, reduction="mean"):
        if not self._knn_done:
            self._calc_knn()

        d1 = (self.a - self._knn1[:, :, 0]).abs().sum(dim=-1)
        d2 = (self.b - self._knn2[:, :, 0]).abs().sum(dim=-1)
        d1 = self._reduce_result(d1, reduction=reduction)
        d2 = self._reduce_result(d2, reduction=reduction)
        return d1 + d2

    def chamfer_l1_legacy(self, reduction="mean"):
        if not self._knn_done:
            self._calc_knn()

        d1 = (self.a - self._knn1[:, :, 0]).square().sum(dim=-1).sqrt()
        d2 = (self.b - self._knn2[:, :, 0]).square().sum(dim=-1).sqrt()
        d1 = self._reduce_result(d1, reduction=reduction)
        d2 = self._reduce_result(d2, reduction=reduction)
        return d1 + d2

    def chamfer_l2(self, reduction="mean"):
        if not self._knn_done:
            self._calc_knn()

        d1 = self._reduce_result(self._dist1, reduction=reduction)
        d2 = self._reduce_result(self._dist2, reduction=reduction)
        return d1 + d2

    def chamfer_lp(self, p: float, reduction="mean"):
        if not self._knn_done:
            self._calc_knn()

        d1 = th.norm(self.a - self._knn1[:, :, 0], p=p, dim=-1)
        d2 = th.norm(self.b - self._knn2[:, :, 0], p=p, dim=-1)
        d1 = self._reduce_result(d1, reduction=reduction)
        d2 = self._reduce_result(d2, reduction=reduction)
        return d1 + d2

    def recall(self, threshold=1e-3):
        if not self._knn_done:
            self._calc_knn()
        return (self._dist1.flatten(1) < threshold).float().mean(1)  # b

    def precision(self, threshold=1e-3):
        if not self._knn_done:
            self._calc_knn()
        return (self._dist2.flatten(1) < threshold).float().mean(1)  # b

    def f1(self, threshold=1e-3, reduction="mean"):
        r = self.recall(threshold)  # b
        p = self.precision(threshold)  # b
        f1 = 2 * r * p / (r + p)  # b
        f1.nan_to_num_(0, 0, 0)

        if reduction == "mean":
            return f1.mean()
        elif reduction == "batchmean":
            return f1
        else:
            raise NotImplementedError(reduction)
