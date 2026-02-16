#!/usr/bin/env python
# coding: utf-8

# This notebook pre-processes the data to be available in the repo path.

# In[1]:


import glob
import json
import os
import pathlib
import shutil
import string

import pandas as pd
import tifffile
from timelapse_utils.file_utils.notebook_init_utils import (
    bandicoot_check,
    init_notebook,
)

root_dir, in_notebook = init_notebook()

if in_notebook:
    import tqdm.notebook as tqdm
else:
    import tqdm


# ## Paths and import data

# In[2]:


bandicoot_mount_path = pathlib.Path(os.path.expanduser("~/mnt/bandicoot/"))
image_base_dir = bandicoot_check(bandicoot_mount_path, root_dir)


# In[3]:


# absolute path to the raw data directory (only works on this machine)
path_to_raw_data = pathlib.Path(
    f"{image_base_dir}/live_cell_timelapse_pyroptosis_project_data/raw_data/"
).resolve(strict=True)

# repository data directory to access the data faster
path_to_processed_data = pathlib.Path(
    f"{image_base_dir}/live_cell_timelapse_pyroptosis_project_data/processed_data/"
).resolve()
path_to_processed_data.mkdir(exist_ok=True, parents=True)

# recurse through the directory and find all the .tif or .tiff files
list_of_files = glob.glob(str(path_to_raw_data / "**/Image/*.tif*"), recursive=True)
list_of_files = sorted(list_of_files)
print(f"Found {len(list_of_files)} files")


# ## Set up a metadata frame

# In[4]:


# make a df out of the file names
df = pd.DataFrame(list_of_files, columns=["file_path"])
df.insert(0, "file_name", df["file_path"].apply(lambda x: pathlib.Path(x).stem))
df.insert(0, "Channel", df["file_name"].apply(lambda x: x.split("C")[1]))
df.insert(
    0, "Time", df["file_name"].apply(lambda x: x.split("Z")[0]).str.split("T").str[1]
)
df.insert(
    0, "FOV", df["file_name"].apply(lambda x: x.split("T")[0]).str.split("F").str[1]
)
df.insert(
    0, "Well", df["file_name"].apply(lambda x: x.split("F")[0]).str.split("W").str[1]
)
df.Channel = df.Channel.astype(int)
df.Time = df.Time.astype(int)
df.FOV = df.FOV.astype(int)
df.Well = df.Well.astype(int)


# In[5]:


# well dictionary for mapping
# Generate the dictionary dynamically
# implemented via Jenna Tomkinson
well_map = {
    i: f"{row}{col}"
    for i, (row, col) in enumerate(
        ((r, c) for r in string.ascii_uppercase[:16] for c in range(1, 25)), start=1
    )
}

# write the well map to a json file
path_to_plate_json_data = pathlib.Path("../../../data/processed/").resolve()
path_to_plate_json_data.mkdir(exist_ok=True, parents=True)
with open(path_to_plate_json_data / "well_map.json", "w") as f:
    json.dump(well_map, f)
# map the well to the well_map
df["Well"] = df["Well"].map(well_map)
# sort by Data_Time
df.sort_values(by=["Well", "FOV", "Time", "Channel"], inplace=True)
df.head(10)


# In[6]:


# rename the processed files to match the new naming convention
for index, row in tqdm.tqdm(df.iterrows(), total=df.shape[0]):
    # skip time 103 as not all wells have this "final time point"
    if row["Time"] == 103:
        continue
    file_path = pathlib.Path(row["file_path"])
    new_file_name = pathlib.Path(
        f"{row['Well']}_{row['FOV']}_T{row['Time']}_C{row['Channel']}.tiff"
    )
    well_fov = f"{row['Well']}_{row['FOV']}"
    new_file_path = path_to_processed_data / well_fov / new_file_name
    new_file_path.parent.mkdir(exist_ok=True, parents=True)
    if not new_file_path.exists():
        shutil.copy(file_path, new_file_path)


# ## File count validations

# In[7]:


# check that there are
# 5 fovs * 5 channels * 96 wells = 2400 images per plate
fovs = 4
channels = 5
wells = 56
time_interval_hours = 0.25
time_elapsed_hours = 20
expected_time_points = int(time_elapsed_hours / time_interval_hours) + 1
images_per_well_fov = channels * expected_time_points
images_per_plate = fovs * channels * wells * expected_time_points

# get the dirs in the data directory
dirs = glob.glob(str(path_to_processed_data / "*"), recursive=False)
dirs = [x for x in dirs if pathlib.Path(x).is_dir()]
plate_dict = {
    "well_fov": [],
    "file_name": [],
}
for dir in tqdm.tqdm(dirs):

    # get the files in the dir
    files = glob.glob(str(pathlib.Path(dir) / "*"))
    files = [x for x in files if pathlib.Path(x).is_file()]
    for file in files:
        plate_dict["well_fov"].append(pathlib.Path(dir).name)
        plate_dict["file_name"].append(file)
plate_df = pd.DataFrame(plate_dict)


# In[8]:


plate_df["file_name"] = plate_df["file_name"].apply(lambda x: pathlib.Path(x).stem)
plate_df["Time"] = (
    plate_df["file_name"].apply(lambda x: x.split("_")[2].split("T")[1]).astype(int)
)
plate_df["Channel"] = (
    plate_df["file_name"].apply(lambda x: x.split("_")[3].split("C")[1]).astype(int)
)


# In[9]:


well_channel_grouped = (
    plate_df.groupby(["well_fov", "Channel"]).size().reset_index(name="num_files")
)
# find cases where num_files is not equal to 103 (number of time points)
expected_time_points = 102
if (
    len(well_channel_grouped[well_channel_grouped["num_files"] != expected_time_points])
    > 0
):
    print(
        well_channel_grouped[well_channel_grouped["num_files"] != expected_time_points]
    )
else:
    print("All well_fov and channel combinations have 102 time points")


# In[10]:


well_time_grouped = (
    plate_df.groupby(["well_fov", "Time"]).size().reset_index(name="num_files")
)
# find cases where num_files is not equal to 5 (number of channels)
if len(well_time_grouped[well_time_grouped["num_files"] != 5]) > 0:
    print(well_time_grouped[well_time_grouped["num_files"] != 5])
else:
    print("All well_fov and time combinations have 5 channels")


# ## File corruption checks

# In[11]:


plate_dict = {
    "well_fov": [],
    "file_name": [],
    "file_size": [],
}
for dir in dirs:
    # get the files in the dir
    files = glob.glob(str(pathlib.Path(dir) / "*"))
    files = [x for x in files if pathlib.Path(x).is_file()]
    for file in files:
        plate_dict["well_fov"].append(pathlib.Path(dir).name)
        plate_dict["file_name"].append(pathlib.Path(file).name)
        plate_dict["file_size"].append(
            pathlib.Path(file).stat().st_size / 1024 / 1024
        )  # convert to MB

file_df = pd.DataFrame(plate_dict)
corruption_counter = 0
corruption_issues = []
for well_fov in file_df["well_fov"].unique():
    subset = file_df[file_df["well_fov"] == well_fov]
    # calculate the mean and std of the file sizes for this well_fov
    mean_file_size = subset["file_size"].mean()
    std_file_size = subset["file_size"].std()
    # if the file size of each file are not all the same, then there is likely a file corruption issue
    if (
        std_file_size > 0.1
    ):  # if the std is greater than 0.1 MB, then there is likely a file corruption issue
        corruption_issues.append(well_fov)
        corruption_counter += 1
print(f"Found {corruption_counter} potential file corruption issues")
if corruption_counter > 0:
    print("The following well_fovs have potential file corruption issues:")
    for issue in corruption_issues:
        print(issue)
