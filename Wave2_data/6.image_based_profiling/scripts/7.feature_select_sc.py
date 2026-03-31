#!/usr/bin/env python
# coding: utf-8

# # Perform feature selection on normalized data

# ## Import libraries

# In[1]:


import os
import pathlib

import pandas as pd
from pycytominer import feature_select
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

feature_selected_profiles_path = pathlib.Path(
    f"{image_base_dir}/9.feature_selected_profiles/feature_selected_profiles.parquet"
).resolve()
feature_selected_profiles_path.parent.mkdir(exist_ok=True)


# ## Define dict of paths

# ## Perform feature selection

# In[3]:


# define operations to be performed on the data
# list of operations for feature select function to use on input profile
feature_select_ops = [
    "variance_threshold",
    "blocklist",
    "drop_na_columns",
    "correlation_threshold",
]


# This last cell does not get run due to memory constraints.
# It is run on an HPC cluster with more memory available.

# In[4]:


# read in the annotated file
normalized_df = pd.read_parquet(normalized_profiles_path)
metadata_cols = [x for x in normalized_df.columns if x.startswith("Metadata_")]
normalized_features_df = normalized_df.drop(metadata_cols, axis="columns")
# perform feature selection with the operations specified
feature_select_df = feature_select(
    normalized_features_df,
    operation=feature_select_ops,
)

# add metadata columns back to the feature selected df
feature_select_df = pd.concat(
    [normalized_df[metadata_cols], feature_select_df], axis="columns"
)
print("Feature selection complete, saving to parquet file!")
# save features selected df as parquet file
output(
    df=feature_select_df,
    output_filename=feature_selected_profiles_path,
    output_type="parquet",
)
# check to see if the shape of the df has changed indicating feature selection occurred
print(feature_select_df.shape)
feature_select_df.head()
