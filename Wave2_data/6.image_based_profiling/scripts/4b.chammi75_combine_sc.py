#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import pathlib
import urllib.request

import matplotlib.pyplot as plt
import natsort
import numpy as np
import pandas as pd
import psutil
import skimage
import tifffile
from timelapse_utils.file_utils.notebook_init_utils import (
    bandicoot_check,
    init_notebook,
)

root_dir, in_notebook = init_notebook()
image_base_dir = bandicoot_check(
    pathlib.Path(os.path.expanduser("~/mnt/bandicoot")).resolve(), root_dir
)

if in_notebook:
    import tqdm.notebook as tqdm
else:
    import tqdm


# In[ ]:


image_base_dir = bandicoot_check(
    root_dir=root_dir,
    bandicoot_mount_path=pathlib.Path(os.path.expanduser("~/mnt/bandicoot")).resolve(),
)


feature_extracted_dir = pathlib.Path(
    image_base_dir
    / "live_cell_timelapse_pyroptosis_project_data"
    / "processed_data"
    / "7a.CHAMMI75_extracted_features"
).resolve(strict=True)

chammi_profiles_file_path = pathlib.Path(
    image_base_dir
    / "live_cell_timelapse_pyroptosis_project_data"
    / "processed_data"
    / "7a.CHAMMI75_extracted_features"
    / "chammi75_combined_sc_profiles.parquet"
).resolve()


# In[3]:


# get a list of each channel to featurize
parquet_dirs = natsort.natsorted(
    [
        x
        for x in feature_extracted_dir.glob("*")
        if x.is_dir() and "run_stats" not in x.name
    ]
)

parquet_files = [file for dir in parquet_dirs for file in dir.glob("*.parquet")]


# In[4]:


df = pd.concat([pd.read_parquet(file) for file in parquet_files], ignore_index=True)
df.to_parquet(chammi_profiles_file_path, index=False)
df.head()
