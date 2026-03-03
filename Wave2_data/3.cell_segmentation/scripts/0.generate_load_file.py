#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import pathlib

import natsort

# Import dependencies
import numpy as np
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


image_base_dir = bandicoot_check(
    root_dir=root_dir,
    bandicoot_mount_path=pathlib.Path(os.path.expanduser("~/mnt/bandicoot")).resolve(),
)

input_dir = pathlib.Path(
    image_base_dir
    / "live_cell_timelapse_pyroptosis_project_data"
    / "processed_data"
    / "1.illumination_corrected_files"
).resolve(strict=True)

segmentation_mask_output_dir = pathlib.Path(
    image_base_dir
    / "live_cell_timelapse_pyroptosis_project_data"
    / "processed_data"
    / "2.cell_segmentation_masks"
).resolve()

loadfile_dir = pathlib.Path("../loadfiles/loadfile.txt").resolve()
loadfile_dir.parent.mkdir(parents=True, exist_ok=True)

EXPECTED_MASK_FILE_COUNT = 102


# ## Set up images, paths and functions

# In[3]:


image_extensions = {".tif", ".tiff"}
raw_image_files = sorted(input_dir.glob("*"))
well_fov_dirs = [x.name for x in raw_image_files if x.is_dir()]


# In[4]:


raw_images_present = {}
nuclei_df_list = []
cell_df_list = []

image_extensions = {".tif", ".tiff"}
raw_image_files = sorted(input_dir.glob("*"))
for well_fov_name in tqdm.tqdm(
    well_fov_dirs, desc="Checking for raw images in input directory"
):
    well_fov_path = segmentation_mask_output_dir / well_fov_name
    if not well_fov_path.exists():
        # add a fake entry with a placeholder file to trigger the reprocessing of this well_fov
        raw_images_present[well_fov_name] = {
            "nuclei": [pathlib.Path("placeholder_nuclei_file.tif")],
            "cell": [pathlib.Path("placeholder_cell_file.tif")],
        }
        continue
    nuclei_files = [
        f
        for f in natsort.natsorted(well_fov_path.glob("*"))
        if f.suffix.lower() in image_extensions and "nuclei" in f.stem.lower()
    ]
    cell_files = [
        f
        for f in natsort.natsorted(well_fov_path.glob("*"))
        if f.suffix.lower() in image_extensions and "cell" in f.stem.lower()
    ]
    raw_images_present[well_fov_name] = {"nuclei": nuclei_files, "cell": cell_files}


for key, value in raw_images_present.items():
    nuclei_df_list.append(
        pd.DataFrame({"well_fov": key, "nuclei_file": value["nuclei"]})
    )
    cell_df_list.append(pd.DataFrame({"well_fov": key, "cell_file": value["cell"]}))
nuclei_df = pd.concat(nuclei_df_list, ignore_index=True)
cell_df = pd.concat(cell_df_list, ignore_index=True)
nuclei_df.head()


# In[5]:


# find the well_fovs with less than the expected number of files
wells_to_rerun = natsort.natsorted(
    set(
        (
            nuclei_df.loc[
                nuclei_df.groupby("well_fov")["nuclei_file"].transform("count")
                < EXPECTED_MASK_FILE_COUNT,
                "well_fov",
            ].unique()
        )
    )
    | set(
        (
            cell_df.loc[
                cell_df.groupby("well_fov")["cell_file"].transform("count")
                < EXPECTED_MASK_FILE_COUNT,
                "well_fov",
            ].unique()
        )
    )
)
if len(wells_to_rerun) == 0:
    print(
        "All well_fovs have the expected number of mask files. No reprocessing needed."
    )
else:
    print(
        f"{len(wells_to_rerun)} well_fovs have less than the expected number of mask files and will be reprocessed. Here are the first 5: {wells_to_rerun[:5]}"
    )


# In[6]:


# write the loadfile
with open(loadfile_dir, "w") as f:
    for well_fov in wells_to_rerun:
        f.write(f"{well_fov}\n")
