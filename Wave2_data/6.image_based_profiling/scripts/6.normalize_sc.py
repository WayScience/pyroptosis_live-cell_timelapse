#!/usr/bin/env python
# coding: utf-8

# # Normalize annotated single cells using negative control

# ## Import libraries

# In[1]:


import os
import pathlib

import pandas as pd
from pycytominer import normalize
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
annotated_profiles_path = pathlib.Path(
    f"{image_base_dir}/7.annotated_profiles/annotated_profiles.parquet"
).resolve(strict=True)

normalized_profiles_path = pathlib.Path(
    f"{image_base_dir}/8.normalized_profiles/normalized_profiles.parquet"
).resolve()
normalized_profiles_path.parent.mkdir(exist_ok=True)


# ## Normalize with standardize method with negative control on annotated data

# The normalization needs to occur per time step.
# This code cell will split the data into time steps and normalize each time step separately.
# Then each normalized time step will be concatenated back together.

# This last cell does not get run due to memory constraints.
# It is run on an HPC cluster with more memory available.

# In[5]:


# set the metadata conditions to fit and apply normalization to
samples = "Metadata_Inducer == 'DMSO' & Metadata_Inducer_dose == '0.15%' & Metadata_Inhibitor == 'DMSO' & Metadata_Inhibitor_dose == '0.15%' & Metadata_Time == '1'"


# In[6]:


# read in the annotated file
annotated_df = pd.read_parquet(annotated_profiles_path)
# get the features (not the metadata) to use for normalization
features = [col for col in annotated_df.columns if "metadata" not in col.lower()]
# apply normalization to the annotated df using the specified samples as the reference for normalization
normalized_df = normalize(
    # df with annotated raw merged single cell features
    profiles=annotated_df,
    features=features,
    # specify samples used as normalization reference (negative control)
    samples=samples,
    # normalization method used
    method="standardize",
)
# save the normalized profiles as a parquet file
output(
    normalized_df,
    output_filename=normalized_profiles_path,
    output_type="parquet",
)
if annotated_df.shape[0] != normalized_df.shape[0]:
    raise ValueError(
        f"Number of rows in the annotated df ({annotated_df.shape[0]}) does not match the number of rows in the normalized df ({normalized_df.shape[0]}). Please check the input annotated df and the samples used for normalization."
    )
normalized_df.head()
