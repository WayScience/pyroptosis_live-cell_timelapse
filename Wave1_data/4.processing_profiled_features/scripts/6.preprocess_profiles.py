#!/usr/bin/env python
# coding: utf-8

# This notebook preprocesses the data to have correct time and treatment metadata.

# In[1]:


import argparse
import json
import pathlib
from pprint import pprint

import pandas as pd

# In[2]:


# check if in a jupyter notebook
try:
    cfg = get_ipython().config
    in_notebook = True
except NameError:
    in_notebook = False

if not in_notebook:
    print("Running as script")
    # set up arg parser
    parser = argparse.ArgumentParser(description="Segment the nuclei of a tiff image")

    parser.add_argument(
        "--samples_per_group",
        type=int,
        default=25,
        help="Number of samples per group",
    )

    parser.add_argument(
        "--data_subset",
        action="store_true",
        help="Use a subset of the data",
    )

    args = parser.parse_args()
    samples_per_group = args.samples_per_group
    data_subset = args.data_subset
else:
    print("Running in a notebook")
    data_subset = True
    samples_per_group = 1000


# In[3]:


normalized_dir = pathlib.Path("../data/normalized_data").resolve()
feature_selected_dir = pathlib.Path("../data/feature_selected_data").resolve()
aggregate_dir = pathlib.Path("../data/aggregated").resolve()
preprocessed_dir = pathlib.Path("../data/preprocessed_data").resolve()
timepoint_dir = pathlib.Path("../../../data/processed/time_to_timepoint.json").resolve()
preprocessed_dir.mkdir(exist_ok=True, parents=True)


# In[4]:


input_data_dict = {
    "first_time": {
        "normalized": {
            "input_data": pathlib.Path(
                f"{normalized_dir}/live_cell_pyroptosis_wave1_sc_first_time_norm.parquet"
            ).resolve(),
            "output_data": pathlib.Path(
                f"{preprocessed_dir}/live_cell_pyroptosis_wave1_sc_first_time_norm.parquet"
            ).resolve(),
        },
        "selected": {
            "input_data": pathlib.Path(
                f"{feature_selected_dir}/live_cell_pyroptosis_wave1_sc_first_time_norm_fs.parquet"
            ).resolve(),
            "output_data": pathlib.Path(
                f"{preprocessed_dir}/live_cell_pyroptosis_wave1_sc_first_time_norm_fs.parquet"
            ).resolve(),
        },
        "aggregate_normalized": {
            "input_data": pathlib.Path(
                f"{aggregate_dir}/live_cell_pyroptosis_wave1_first_time_norm_agg.parquet"
            ).resolve(),
            "output_data": pathlib.Path(
                f"{preprocessed_dir}/live_cell_pyroptosis_wave1_first_time_norm_agg.parquet"
            ).resolve(),
        },
        "aggregate_selected": {
            "input_data": pathlib.Path(
                f"{aggregate_dir}/live_cell_pyroptosis_wave1_first_time_norm_fs_agg.parquet"
            ).resolve(),
            "output_data": pathlib.Path(
                f"{preprocessed_dir}/live_cell_pyroptosis_wave1_first_time_norm_fs_agg.parquet"
            ).resolve(),
        },
    },
}

pprint(input_data_dict)


# In[5]:


# load the time map
with open(timepoint_dir, "r") as f:
    time_map = json.load(f)


# This last cell does not get run due to memory constraints.
# It is run on an HPC cluster with more memory available.

# In[ ]:


for dataset in input_data_dict:
    for data_type in input_data_dict[dataset]:
        data = pd.read_parquet(input_data_dict[dataset][data_type]["input_data"])

        # drop Wells N04, N06, N08, and N10 as they have no Hoechst stain
        data = data[~data["Metadata_Well"].str.contains("N04|N06|N08|N10")]
        # map the time to the time point in hours
        data["Metadata_Time"] = data["Metadata_Time"].map(lambda x: time_map[x])

        if "aggregate" in data_type:
            data.to_parquet(input_data_dict[dataset][data_type]["output_data"])
        elif data_subset:
            # sample the data stratified by Metadata_Well and Metadata Time
            data = data.groupby(["Metadata_Well", "Metadata_Time"]).apply(
                lambda x: x.sample(samples_per_group)
            )
            subset_output = (
                input_data_dict[dataset][data_type]["output_data"].parent
                / f"{input_data_dict[dataset][data_type]['output_data'].stem}_subset.parquet"
            )
            data.to_parquet(subset_output)
        else:
            data.to_parquet(input_data_dict[dataset][data_type]["output_data"])

        print(f"Preprocessed data for {dataset} has the shape: {data.shape}")


# In[ ]:


data.head()
