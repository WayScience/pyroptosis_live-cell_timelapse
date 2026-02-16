"""Notebook initialization helpers and Bandicoot path utilities."""

import os
import pathlib
from typing import Tuple


def init_notebook() -> Tuple[pathlib.Path, bool]:
    """
    Description
    -----------
    Initializes the notebook environment by determining the root directory of the Git repository
    and checking if the code is running in a Jupyter notebook.

    Returns
    -------
    Tuple[pathlib.Path, bool]
        - pathlib.Path: The root directory of the Git repository.
        - bool: True if running in a Jupyter notebook, False otherwise.
    """
    try:
        cfg = get_ipython().config
        in_notebook = True
    except NameError:
        in_notebook = False

    # Get the current working directory
    cwd = pathlib.Path.cwd()

    if (cwd / ".git").is_dir():
        root_dir = cwd

    else:
        root_dir = None
        for parent in cwd.parents:
            if (parent / ".git").is_dir():
                root_dir = parent
                break

    # Check if a Git root directory was found
    if root_dir is None:
        raise FileNotFoundError("No Git root directory found.")
    return root_dir, in_notebook


def bandicoot_check(
    bandicoot_mount_path: pathlib.Path, root_dir: pathlib.Path
) -> pathlib.Path:
    """
    This function determines if the external mount point for Bandicoot exists.

    Parameters
    ----------
    bandicoot_mount_path : pathlib.Path
        The path to the Bandicoot mount point.
    root_dir : pathlib.Path
        The root directory of the Git repository.

    Returns
    -------
    pathlib.Path
        The base directory for image data.
    """
    if bandicoot_mount_path.exists():
        # comment out depending on whose computer you are on
        # mike's computer
        image_base_dir = pathlib.Path(os.path.expanduser("~/mnt/bandicoot/")).resolve(
            strict=True
        )
    else:
        image_base_dir = root_dir
    return image_base_dir
