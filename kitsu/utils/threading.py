from multiprocessing import Pool
from multiprocessing.dummy import Pool as ThreadPool
from typing import Any, Callable, List, Sequence, Tuple

__all__ = ["distributed_run"]


def _thread_func(func):
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs), None
        except Exception as e:
            return None, e

    return wrapper


def distributed_run(
    n_thread: int,
    data: Sequence,
    func: Callable[[int, Any], Any],
    callback: Callable[[int, Any], Any] = None,
    multiprocess: bool = False,
):
    pool = (Pool if multiprocess else ThreadPool)(n_thread)
    callback = callback or (lambda i, x: x)

    func = _thread_func(func)
    callback = _thread_func(callback)

    output_ret = [None for _ in range(len(data))]
    output_err = [None for _ in range(len(data))]

    for i, x in enumerate(data):
        ret, err = func(i, x)
        if err is None:
            ret, err = callback(i, ret)

        if err is None:
            output_ret[i] = ret
        else:
            output_err[i] = err

    pool.close()
    return output_ret, output_err
