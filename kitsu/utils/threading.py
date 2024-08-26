from multiprocessing import Pool
from multiprocessing.dummy import Pool as ThreadPool
from typing import Any, Callable, List, Sequence, Tuple

__all__ = ["thread_func"]


def thread_func(func):
    def wrapper(i, *args, **kwargs):
        return i, func(*args, **kwargs)

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

    output = [None for _ in range(len(data))]
    for i, x in enumerate(data):
        output[i] = callback(i, func(i, x))

    pool.close()
    return output
