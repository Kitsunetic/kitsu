import os
import time
from os import PathLike
from pathlib import Path
from typing import Union

__all__ = ["try_remove_file"]


def try_remove_file(file: Union[str, Path, PathLike]):
    for _ in range(10):
        try:
            os.remove(str(file))
            break
        except:
            print("Warn: Failed to remove", str(file))
            time.sleep(0.1)
