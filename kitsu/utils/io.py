import os
import time

__all__ = ["try_remove_file"]


def try_remove_file(file):
    for _ in range(10):
        try:
            os.remove(file)
            break
        except:
            print("Warn: Failed to remove", file)
            time.sleep(0.1)
