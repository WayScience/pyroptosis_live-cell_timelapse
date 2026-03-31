#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import pathlib

import matplotlib.pyplot as plt
import natsort
import numpy as np
import pandas as pd
import seaborn as sns
from pycytominer import annotate
from pycytominer.cyto_utils import output
from timelapse_utils.file_utils.notebook_init_utils import (
    bandicoot_check,
    init_notebook,
)
from timelapse_utils.profiling_utils.sc_extraction_utils import add_single_cell_count_df

root_dir, in_notebook = init_notebook()
if in_notebook:
    import tqdm.notebook as tqdm
else:
    import tqdm


# ## Set paths and variables

# In[2]:


image_base_dir = bandicoot_check(
    bandicoot_mount_path=pathlib.Path(f"{os.path.expanduser('~')}/mnt/bandicoot/"),
    root_dir=root_dir,
)
image_base_dir = pathlib.Path(
    f"{image_base_dir}/live_cell_timelapse_pyroptosis_project_data/processed_data/"
).resolve(strict=True)
ic_image_dir = pathlib.Path(
    f"{image_base_dir}/1.illumination_corrected_files/"
).resolve(strict=True)

load_data_file_path = pathlib.Path(
    f"{root_dir}/Wave2_data/7.image_based_profiling/load_data/load_file.txt"
).resolve()
load_data_file_path.parent.mkdir(exist_ok=True, parents=True)


# In[13]:


# well_fov_timepoints
image_list = [x for x in tqdm.tqdm(ic_image_dir.glob("*")) if x.is_dir()]
image_list = natsort.natsorted(image_list)
image_list = [
    list(x.glob("**/*.tiff"))
    for x in tqdm.tqdm(image_list)
    if len(list(x.glob("**/*.tiff"))) > 0
]
image_list = natsort.natsorted(image_list)
print(f"Number of images: {len(image_list)}")


# In[19]:


# unnest the nested list of lists
image_list = [
    item
    for sublist in image_list
    for item in sublist
    if isinstance(item, list) or not isinstance(item, pathlib.Path)
]
print(f"Number of images after unnesting: {len(image_list)}")


# In[25]:


df = pd.DataFrame({"image_path": image_list})

df = pd.DataFrame({"image_path": image_list})
df["file_name"] = df["image_path"].apply(lambda x: x.stem)
df["well_fov_time"] = df["file_name"].apply(lambda x: "_".join(x.split("_")[:3]))
df["well_fov_time"] = df["well_fov_time"].str.replace("T", "")
well_fov_times = natsort.natsorted(list(set(df["well_fov_time"].to_list())))

with open(load_data_file_path, "w") as f:
    for item in well_fov_times:
        f.write(f"{item}\n")
print(
    f"Saved load data file with {len(well_fov_times)} entries to {load_data_file_path}"
)
