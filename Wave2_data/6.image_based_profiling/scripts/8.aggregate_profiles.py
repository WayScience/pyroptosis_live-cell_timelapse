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
).resolve(strict=True)

norm_agg_profiles_path = pathlib.Path(
    f"{image_base_dir}/10.aggregate_profiles/aggregate_profiles_from_normalized.parquet"
).resolve()
norm_consensus_profiles_path = pathlib.Path(
    f"{image_base_dir}/10.aggregate_profiles/consensus_profiles_from_normalized.parquet"
).resolve()
fs_agg_profiles_path = pathlib.Path(
    f"{image_base_dir}/10.aggregate_profiles/aggregate_profiles_from_feature_selected.parquet"
).resolve()
fs_consensus_profiles_path = pathlib.Path(
    f"{image_base_dir}/10.aggregate_profiles/consensus_profiles_from_feature_selected.parquet"
).resolve()
fs_consensus_profiles_path.parent.mkdir(exist_ok=True)


# This last cell does not get run due to memory constraints.
# It is run on an HPC cluster with more memory available.

# In[3]:


agg_strata = ["Metadata_Well", "Metadata_Time"]
consensus_strata = [
    "Metadata_Inducer",
    "Metadata_Inducer_dose",
    "Metadata_Inhibitor",
    "Metadata_Inhibitor_dose",
    "Metadata_Time",
]


# In[4]:


###########################################################################################
# Normalized aggregate data
###########################################################################################
# Load the normalized data
norm_df = pd.read_parquet(normalized_profiles_path)
metadata_cols = [cols for cols in norm_df.columns if "Metadata" in cols]
features_cols = [cols for cols in norm_df.columns if "Metadata" not in cols]

norm_aggregate_df = aggregate(
    population_df=norm_df,
    strata=agg_strata,
    features=features_cols,
    operation="median",
)
# Drop metadata columns
metadata_df = norm_df[metadata_cols]
metadata_df = metadata_df.drop_duplicates(subset=agg_strata).reset_index(drop=True)

norm_aggregate_df = pd.merge(metadata_df, norm_aggregate_df, on=agg_strata)
# Save the aggregated normalized data
norm_aggregate_df.to_parquet(norm_agg_profiles_path, index=False)

###############################################################################
# Norm consensus profiles
###############################################################################
# Load the normalized data
norm_df = pd.read_parquet(normalized_profiles_path)
metadata_cols = [cols for cols in norm_df.columns if "Metadata" in cols]
features_cols = [cols for cols in norm_df.columns if "Metadata" not in cols]

norm_consensus_df = aggregate(
    population_df=norm_df,
    strata=consensus_strata,
    features=features_cols,
    operation="median",
)
# Drop metadata columns
metadata_df = norm_df[metadata_cols]
metadata_df = metadata_df.drop_duplicates(subset=consensus_strata).reset_index(
    drop=True
)

norm_consensus_df = pd.merge(metadata_df, norm_consensus_df, on=consensus_strata)
# Save the aggregated normalized data
norm_consensus_df.to_parquet(norm_consensus_profiles_path, index=False)

###########################################################################################
# Feature selected aggregate data
###########################################################################################
# Load the feature selected data
feature_select_df = pd.read_parquet(feature_selected_profiles_path)
metadata_cols = [cols for cols in feature_select_df.columns if "Metadata" in cols]
features_cols = [cols for cols in feature_select_df.columns if "Metadata" not in cols]
feature_select_agg_df = aggregate(
    population_df=feature_select_df,
    strata=agg_strata,
    features=features_cols,
    operation="median",
)
# Drop metadata columns
metadata_df = feature_select_df[metadata_cols]
metadata_df = metadata_df.drop_duplicates(subset=agg_strata).reset_index(drop=True)

feature_select_agg_df = pd.merge(metadata_df, feature_select_agg_df, on=agg_strata)
# Save the aggregated normalized data
feature_select_agg_df.to_parquet(fs_agg_profiles_path, index=False)

###############################################################################
# Feature selected consensus profiles
###############################################################################
# Load the feature selected data
feature_select_df = pd.read_parquet(feature_selected_profiles_path)
metadata_cols = [cols for cols in feature_select_df.columns if "Metadata" in cols]
features_cols = [cols for cols in feature_select_df.columns if "Metadata" not in cols]
feature_select_consensus_df = aggregate(
    population_df=feature_select_df,
    strata=consensus_strata,
    features=features_cols,
    operation="median",
)

# Drop metadata columns
metadata_df = feature_select_df[metadata_cols]
metadata_df = metadata_df.drop_duplicates(subset=consensus_strata).reset_index(
    drop=True
)
feature_select_consensus_df = pd.merge(
    metadata_df, feature_select_consensus_df, on=consensus_strata
)
# Save the aggregated normalized data
feature_select_consensus_df.to_parquet(fs_consensus_profiles_path, index=False)

print(f"Normalized df shape: {norm_df.shape}")
print(f"Normalized aggregate df shape: {norm_aggregate_df.shape}")
print(f"Normalized consensus df shape: {norm_consensus_df.shape}")
print(f"Feature selected df shape: {feature_select_df.shape}")
print(f"Feature selected aggregate df shape: {feature_select_agg_df.shape}")
print(f"Feature selected consensus df shape: {feature_select_consensus_df.shape}")
