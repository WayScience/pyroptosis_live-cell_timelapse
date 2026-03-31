#!/usr/bin/env python
# coding: utf-8

# # Aggregate the single-cell profiles to the well level
# This notebook is not run as a large amount of RAM is needed to run it. It is provided for reference only.

# In[1]:


import os
import pathlib

import pandas as pd
from pycytominer import aggregate
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


# In[4]:


# load in platemap file as a pandas dataframe
platemap_path = pathlib.Path(
    f"{root_dir}/Wave2_data/0.download_data/platemap/platemap.csv"
).resolve()

image_base_dir = bandicoot_check(
    bandicoot_mount_path=pathlib.Path(f"{os.path.expanduser('~')}/mnt/bandicoot/"),
    root_dir=root_dir,
)
image_base_dir = pathlib.Path(
    f"{image_base_dir}/live_cell_timelapse_pyroptosis_project_data/processed_data/"
).resolve(strict=True)
normalized_profiles_path = pathlib.Path(
    f"{image_base_dir}/8.normalized_profiles/normalized_profiles.parquet"
).resolve(strict=True)
chammi_profiles_path = pathlib.Path(
    f"{image_base_dir}/7a.CHAMMI75_extracted_features/CHAMMI75_combined_sc_profiles.parquet"
).resolve(strict=True)


# In[5]:


norm_CP_df = pd.read_parquet(normalized_profiles_path)
chammi75_df = pd.read_parquet(chammi_profiles_path)
print("Shape of profiles:")
print(f"Normalized CP: {norm_CP_df.shape[0]}")
print(f"CHAMMI75: {chammi75_df.shape[0]}")


# In[7]:


norm_CP_df


# In[8]:


merged_df = pd.merge(
    norm_CP_df,
    chammi75_df,
    on=[
        "Metadata_Well_FOV",
        "Metadata_Time",
        "Metadata_Nuclei_Number_Object_Number",
        "Metadata_Cells_AreaShape_Center_X",
        "Metadata_Cells_AreaShape_Center_Y",
    ],
    how="left",
)
print(f"Merged profiles: {merged_df.shape[0]}")


# In[10]:


merged_df


# In[ ]:
