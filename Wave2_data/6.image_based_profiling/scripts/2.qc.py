#!/usr/bin/env python
# coding: utf-8

# # Annotate merged single cells with metadata from platemap file

# ## Import libraries

# In[1]:


import os
import pathlib

import cosmicqc
import matplotlib.pyplot as plt
import natsort
import numpy as np
import pandas as pd
import seaborn as sns
from cytodataframe import CytoDataFrame
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
combined_profiles_path = (
    image_base_dir / "5.combined_profiles" / "combined_profiles.parquet"
)
qc_profiles_path = image_base_dir / "6.qc_profiles" / "qc_profiles.parquet"
qc_profiles_path.parent.mkdir(parents=True, exist_ok=True)


# In[3]:


df = pd.read_parquet(combined_profiles_path)
metadata_cols = [x for x in df.columns if "Metadata_" in x]
feature_cols = [x for x in df.columns if "Metadata_" not in x]
features_of_interest = [
    "Nuclei_AreaShape_Area",
    "Nuclei_AreaShape_FormFactor",
    "Nuclei_AreaShape_Eccentricity",
]
df_merged_single_cells = df.copy()
df_merged_single_cells = df[metadata_cols + features_of_interest]


# In[4]:


df_labeled_outliers = cosmicqc.analyze.find_outliers(
    df=df_merged_single_cells,
    metadata_columns=metadata_cols,
    feature_thresholds="large_nuclei",
)
df_labeled_outliers = cosmicqc.analyze.find_outliers(
    df=df_merged_single_cells,
    metadata_columns=metadata_cols,
    feature_thresholds="elongated_nuclei",
)
df_labeled_outliers = cosmicqc.analyze.find_outliers(
    df=df_merged_single_cells,
    metadata_columns=metadata_cols,
    feature_thresholds="small_and_low_formfactor_nuclei",
)


# In[5]:


df_labeled_outliers = cosmicqc.analyze.label_outliers(
    df=df_merged_single_cells,
    include_threshold_scores=True,
)


# In[6]:


# create a column which indicates whether an erroneous outlier was detected
# from all cosmicqc outlier threshold sets. For ex. True for is_outlier in
# one threshold set out of three would show True for this column. False for
# is_outlier in all threshold sets would show False for this column.
df_labeled_outliers["analysis.included_at_least_one_outlier"] = df_labeled_outliers[
    [col for col in df_labeled_outliers.columns.tolist() if ".is_outlier" in col]
].any(axis=1)
df_labeled_outliers = df_labeled_outliers["analysis.included_at_least_one_outlier"]
# show value counts for all outliers
outliers_counts = df_labeled_outliers.value_counts()
outliers_counts


# In[7]:


# show the percentage of total dataset
print(
    np.round((outliers_counts.iloc[1] / outliers_counts.iloc[0]) * 100, 2),
    "%",
    "of",
    outliers_counts.iloc[0],
    "include erroneous outliers of some kind.",
)


# In[8]:


before_shape = df.shape
df = df.iloc[df_labeled_outliers.index[df_labeled_outliers == False], :]
print(
    f"Prior to qc we had {before_shape[0]} rows and after removing outliers we have {df.shape[0]} rows."
)


# In[9]:


df.to_parquet(qc_profiles_path, index=False)


# In[10]:


df.head()
