import pathlib
import re
from typing import Tuple


def natural_key(input_list: list) -> list:
    """
    Generates the natural sorted list from an input list of strings

    Parameters
    ----------
    input_list : list
        A list of strings to be sorted in natural order.

    Returns
    -------
    list
        A list of strings sorted in natural order.
    """
    return [
        int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", input_list[0])
    ]


def t_c_key(p: pathlib.Path) -> Tuple[int, int]:
    """
    This function generates a key for sorting files based on their time (T) and channel (C) indices extracted from the filename. It uses regular expressions to find the T and C values in the filename and returns them as a tuple of integers. If the pattern is not found, it returns a tuple of infinity values to ensure that such files are sorted at the end.

    Parameters
    ----------
    p : pathlib.Path
        A pathlib.Path object representing a file path.

    Returns
    -------
    Tuple[int, int]
        A tuple of integers representing the time and channel indices.
    """
    m = re.search(r"_T(\d+)_C(\d+)\.", p.name)
    return (int(m.group(1)), int(m.group(2))) if m else (float("inf"), float("inf"))
