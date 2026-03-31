#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import argparse
import os
import pathlib
import sys
import urllib.request

import matplotlib.pyplot as plt
import natsort
import numpy as np
import pandas as pd
import psutil
import skimage
import tifffile
import torch
from timelapse_utils.featurization_utils.chammi75_featurization import (
    PerImageNormalize,
    SaturationNoiseInjector,
    call_chammi75_featurization_pipeline,
    featurize_2D_image_w_chammi75,
    get_chammi75_model,
)
from timelapse_utils.featurization_utils.feature_writing_utils import (
    format_morphology_feature_name,
)
from timelapse_utils.featurization_utils.resource_profiling import (
    start_profiling,
    stop_profiling,
)
from timelapse_utils.file_utils.notebook_init_utils import (
    bandicoot_check,
    init_notebook,
)
from timelapse_utils.image_utils.timelapse_image_utils import (
    change_bbox_dtype_to_integer,
    check_for_xy_squareness,
    crop_from_centroid,
    extract_x_y_centroid_from_image_based_profile,
    square_off_xy_crop_bbox,
)

root_dir, in_notebook = init_notebook()
image_base_dir = bandicoot_check(
    pathlib.Path(os.path.expanduser("~/mnt/bandicoot")).resolve(), root_dir
)

if in_notebook:
    import tqdm.notebook as tqdm
else:
    import tqdm


# In[2]:


start_time, start_memory = start_profiling()


# In[3]:


if not in_notebook:
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "--well_fov_time",
        type=str,
        help="The well, fov, and timepoint to featurize in the format 'well_fov_timepoint', e.g. 'A1_1_1'",
    )
    args = argparser.parse_args()
    well_fov_time = args.well_fov_time
else:
    well_fov_time = "B2_1_1"

well = well_fov_time.split("_")[0]
fov = well_fov_time.split("_")[1]
timepoint = well_fov_time.split("_")[2]
well_fov = f"{well}_{fov}"


image_base_dir = bandicoot_check(
    root_dir=root_dir,
    bandicoot_mount_path=pathlib.Path(os.path.expanduser("~/mnt/bandicoot")).resolve(),
)

profile_file_path = pathlib.Path(
    image_base_dir
    / "live_cell_timelapse_pyroptosis_project_data"
    / "processed_data"
    / "7.annotated_profiles"
    / "annotated_profiles.parquet"
).resolve(strict=True)

channel_image_dir = pathlib.Path(
    image_base_dir
    / "live_cell_timelapse_pyroptosis_project_data"
    / "processed_data"
    / "1.illumination_corrected_files"
    / well_fov
).resolve(strict=True)

feature_extracted_file = pathlib.Path(
    image_base_dir
    / "live_cell_timelapse_pyroptosis_project_data"
    / "processed_data"
    / "7a.CHAMMI75_extracted_features"
    / well_fov_time
    / "chammi75_features.parquet"
).resolve()
feature_extracted_file.parent.mkdir(exist_ok=True, parents=True)

if feature_extracted_file.exists():
    # exit the script/notebook we already have these features extracted
    sys.exit(
        f"Features already extracted for {well_fov_time} at {feature_extracted_file}. Exiting."
    )


# In[4]:


channel_mapping = {
    "C1": "ChomaLIVE561",
    "C2": "ChomaLIVE488",
    "C3": "SYTOXGreen",
    "C4": "NucleoLIVE",
    "C5": "Brightfield",
}


# In[5]:


# get a list of each channel to featurize
channel_paths = natsort.natsorted(
    list(channel_image_dir.glob(f"{well_fov}_T{timepoint}_C*.tiff"))
)
channel_images = {
    channel_mapping[path.stem.split("_")[-2]]: tifffile.imread(path)
    for path in channel_paths
}


# In[6]:


device = "cuda" if torch.cuda.is_available() else "cpu"
chammi75_model = get_chammi75_model(device)


# In[7]:


columns_to_read_in = [
    "Metadata_Well_FOV_Time",
    "Metadata_Cells_AreaShape_Center_X",
    "Metadata_Cells_AreaShape_Center_Y",
    "Metadata_Nuclei_Number_Object_Number",
    "Metadata_Cells_Number_Object_Number",
]


# In[8]:


annotated_df = pd.read_parquet(profile_file_path, columns=columns_to_read_in)
# subset the dataframe to just the row corresponding to the well_fov_time we're featurizing
annotated_df = annotated_df.loc[annotated_df["Metadata_Well_FOV_Time"] == well_fov_time]


# In[9]:


label_ids = np.unique(annotated_df["Metadata_Nuclei_Number_Object_Number"])
print(f"Found {len(label_ids)} unique label IDs")
# read in the first channel to get the image dimensions for cropping
image_shape = channel_images["NucleoLIVE"].shape


# In[10]:


list_of_feature_dicts = []
for label in tqdm.tqdm(label_ids, desc="Extracting features for objects", leave=True):
    tmp_df = annotated_df.loc[
        annotated_df["Metadata_Nuclei_Number_Object_Number"] == label
    ]
    tmp_df.reset_index(inplace=True, drop=True)
    center_x, center_y = extract_x_y_centroid_from_image_based_profile(
        tmp_df,
        label=label,
        label_column_name="Metadata_Nuclei_Number_Object_Number",
    )
    bbox = crop_from_centroid(
        center_x=center_x,
        center_y=center_y,
        image_shape=image_shape,
        radius=30,
    )
    # loop through the channel images to crop and featurize each one
    for channel_name, channel_image in channel_images.items():
        # crop the original image to the new bbox
        cropped_image = channel_image[bbox[0] : bbox[2], bbox[1] : bbox[3]]

        chammi75_features = call_chammi75_featurization_pipeline(
            cropped_image=cropped_image, model=chammi75_model
        )
        # make a new dictionary to hold all features
        combined_feature_dict = {
            "feature_name": [],
            "feature_value": [],
            "Metadata_Nuclei_Number_Object_Number": [],
            "Metadata_Well_FOV": [],
            "Metadata_Time": [],
            "Metadata_Cells_AreaShape_Center_X": [],
            "Metadata_Cells_AreaShape_Center_Y": [],
        }

        for i, feat_value in enumerate(chammi75_features[0]):
            combined_feature_dict["feature_name"].append(
                format_morphology_feature_name(
                    compartment="Nucleocentric",
                    channel=channel_name,
                    feature_type="CHAMMI75",
                    measurement=f"Feature{i}",
                )
            )
            combined_feature_dict["feature_value"].append(feat_value.item())
            combined_feature_dict["Metadata_Nuclei_Number_Object_Number"].append(label)
            combined_feature_dict["Metadata_Well_FOV"].append(well_fov)
            combined_feature_dict["Metadata_Time"].append(timepoint)
            combined_feature_dict["Metadata_Cells_AreaShape_Center_X"].append(
                tmp_df.iloc[0]["Metadata_Cells_AreaShape_Center_X"]
            )
            combined_feature_dict["Metadata_Cells_AreaShape_Center_Y"].append(
                tmp_df.iloc[0]["Metadata_Cells_AreaShape_Center_Y"]
            )
        df = pd.DataFrame(combined_feature_dict)
        list_of_feature_dicts.append(df)


# In[11]:


final_df = pd.concat(list_of_feature_dicts, ignore_index=True)
# pivot the df such that the feature names are the columns and the feature values are the values, with object label as an id variable
final_df = final_df.pivot(
    index=[
        "Metadata_Nuclei_Number_Object_Number",
        "Metadata_Well_FOV",
        "Metadata_Time",
        "Metadata_Cells_AreaShape_Center_X",
        "Metadata_Cells_AreaShape_Center_Y",
    ],
    columns="feature_name",
    values="feature_value",
).reset_index()
# remove the labeld name of the index
final_df.columns.name = None
final_df["Metadata_Nuclei_Number_Object_Number"] = final_df[
    "Metadata_Nuclei_Number_Object_Number"
].astype(int)
final_df.to_parquet(feature_extracted_file)
final_df.head()


# In[12]:


run_stats_path = pathlib.Path(
    image_base_dir
    / "live_cell_timelapse_pyroptosis_project_data"
    / "processed_data"
    / "7a.CHAMMI75_extracted_features"
    / "run_stats"
    / f"{well_fov_time}_chammi75_featurization_run_stats.parquet"
).resolve()
run_stats_path.parent.mkdir(exist_ok=True, parents=True)


# In[13]:


stop_profiling(
    start_time=start_time,
    well_fov=well_fov,
    timepoint=timepoint,
    feature_type="CHAMMI75",
    channel="All",
    compartment="Nucleocentric",
    CPU_GPU=str(device),
    output_file_dir=run_stats_path,
    start_mem=start_memory,
)
