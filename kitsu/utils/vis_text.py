import re
from typing import Dict, List

__all__ = ["make_table", "convert_file_size", "parse_file_size", "convert_dates"]


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
