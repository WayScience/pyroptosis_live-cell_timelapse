#!/usr/bin/env python
# coding: utf-8

# In[1]:


import argparse
import os
import pathlib

import natsort

# Import dependencies
import pandas as pd
from timelapse_utils.file_utils.notebook_init_utils import (
    bandicoot_check,
    init_notebook,
)

root_dir, in_notebook = init_notebook()
if in_notebook:
    import tqdm.notebook as tqdm
else:
    import tqdm


# In[2]:


if not in_notebook:
    args = argparse.ArgumentParser()
    args.add_argument(
        "--plate_name",
        type=str,
        required=True,
        help="Name of the plate to process (e.g. '2023-08-01_plate1')",
    )
    plate_name = args.parse_args().plate_name
else:
    plate_name = "plate_2"


# In[3]:


image_base_dir = bandicoot_check(
    root_dir=root_dir,
    bandicoot_mount_path=pathlib.Path(os.path.expanduser("~/mnt/bandicoot")).resolve(),
)
segmentation_mask_dir = pathlib.Path(
    image_base_dir / "processed_data" / "2.cell_segmentation_masks" / plate_name
).resolve()
cell_tracks_dir = pathlib.Path(
    image_base_dir / "processed_data" / "3.cell_tracks" / plate_name
).resolve()
loadfile_dir = pathlib.Path("../loadfiles/loadfile.txt").resolve()
loadfile_dir.parent.mkdir(parents=True, exist_ok=True)


# ## Set up images, paths and functions

# In[4]:


image_extensions = {".tif", ".tiff"}
segmentation_mask_dirs = sorted(segmentation_mask_dir.glob("*"))
well_fov_dirs = [x.name for x in segmentation_mask_dirs if x.is_dir()]


# In[5]:


cell_tracks_to_run = []

for well_fov_name in tqdm.tqdm(
    well_fov_dirs, desc="Checking for raw images in input directory"
):
    well_fov_path = cell_tracks_dir / well_fov_name
    if not well_fov_path.exists():
        cell_tracks_to_run.append(well_fov_name)
df = pd.DataFrame(cell_tracks_to_run, columns=["well_fov"])
wells_to_rerun = [x for x in df["well_fov"].unique()]


# In[6]:


# find the well_fovs with less than the expected number of files
if len(wells_to_rerun) == 0:
    print(
        "All well_fovs have the expected number of mask files. No reprocessing needed."
    )
else:
    if len(wells_to_rerun) < 5:
        n = len(wells_to_rerun)
    else:
        n = 5
    print(
        f"{len(wells_to_rerun)} well_fovs have less than the expected number of mask files and will be reprocessed. Here are the first {n}: {wells_to_rerun[:n]}"
    )


# In[7]:


# write the loadfile
with open(loadfile_dir, "w") as f:
    for well_fov in wells_to_rerun:
        f.write(f"{well_fov}\n")
