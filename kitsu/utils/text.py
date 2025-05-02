import re
from typing import Dict, List

__all__ = ["make_table", "convert_file_size", "parse_file_size", "convert_dates", "print_dict"]


def make_table(df: Dict[str, List[str]]) -> str:
    """Turn the given `df` (dataframe) into a table that is made up with only `-`, `|`, and ` `.
    Args:
        df (Dict[str, List[str]]):
    """
    max_lens = {k: max([len(str(k))] + [len(str(x)) for x in df[k]]) for k in df}
    W = sum(list(max_lens.values())) + 3 * (len(max_lens) - 1)  # width

    n_items = [len(df[k]) for k in df]
    N = n_items[0]
    assert all([n_item == N for n_item in n_items]), f"Every column should have same number of items but {n_items}"

    out = "-" * W + "\n"
    out += " | ".join([("{:<%d}" % max_lens[k]).format(str(k)) for k in df]) + "\n"
    out += "-" * W + "\n"
    for i in range(N):
        out += " | ".join([("{:<%d}" % max_lens[k]).format(str(df[k][i])) for k in df]) + "\n"
    out += "-" * W + "\n"
    return out[:-1]


def convert_file_size(size: float, format: str = "%.2f") -> str:
    if size > 1024**4:
        return format % (size / 1024**4) + "TB"
    elif size > 1024**3:
        return format % (size / 1024**3) + "GB"
    elif size > 1024**2:
        return format % (size / 1024**2) + "MB"
    elif size > 1024:
        return format % (size / 1024) + "KB"
    else:
        return format % size + "B"


def parse_file_size(size: str) -> int:
    pattern = re.compile(r"(\d*\.?\d+)\s*(B|KB|MB|GB|TB)", re.IGNORECASE)
    search = pattern.search(size)
    if search:
        group = search.groups()
        val = float(group[0])
        unit = 1024 ** ({"B": 0, "KB": 1, "MB": 2, "GB": 3, "TB": 4}[group[1].upper()])
        return round(val * unit)
    else:
        return 0


def convert_dates(date: int) -> str:
    if date > 365:
        return str(date // 365) + " years"
    elif date > 30:
        return str(date // 30) + " months"
    elif date > 7:
        return str(date // 7) + " weeks"
    else:
        return str(date)


def print_dict(dict: dict):
    from typing import Dict, List, Set, Tuple

    import numpy as np
    from torch import Tensor

    keylens = [len(k) for k in dict.keys()]
    max_keylen = max(keylens)

    for k, v in dict.items():
        if isinstance(v, (Tensor, np.ndarray)):
            print(("{:<%d}: {}, {}" % max_keylen).format(k, v.shape, v.dtype))
        elif isinstance(v, (List, Tuple, Set)):
            if isinstance(v, List):
                beg, end = "[]"
            elif isinstance(v, Tuple):
                beg, end = "()"
            else:
                beg, end = "\{\}"

            if len(v) == 0:
                print(("{:<%d}: {}{}" % max_keylen).format(k, beg, end))
            else:
                dots = ", ..." if len(v) > 4 else ""
                items = ", ".join(map(str, v[:4]))
                print(("{:<%d}: {}{}{}{}" % max_keylen).format(k, beg, items, dots, end))
        elif isinstance(v, Dict):
            dots = ", ..." if len(dict) > 1 else ""
            for j, u in v.items():
                print(("{:<%d}: {{{}: {}{}}}" % max_keylen).format(k, j, u, dots))
                break
        else:
            print(("{:<%d}: {}" % max_keylen).format(k, v))
