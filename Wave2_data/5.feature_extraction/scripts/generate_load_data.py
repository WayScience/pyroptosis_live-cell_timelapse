#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import csv
import os
import pathlib
import re
import time

from timelapse_utils.file_utils.notebook_init_utils import (
    bandicoot_check,
    init_notebook,
)

root_dir, in_notebook = init_notebook()
if in_notebook:
    import tqdm.notebook as tqdm
else:
    import tqdm

image_based_dir = bandicoot_check(
    bandicoot_mount_path=pathlib.Path(f"{os.path.expanduser('~')}/mnt/bandicoot/"),
    root_dir=root_dir,
)
image_based_dir = image_based_dir / "live_cell_timelapse_pyroptosis_project_data"


# ## Set paths and variables

# In[2]:


fieldnames = [
    "Metadata_Well",
    "Metadata_Time",
    "Metadata_WellFOV",
    "Image_FileName_CL640",
    "Image_PathName_CL640",
    "Image_FileName_CL488",
    "Image_PathName_CL488",
    "Image_FileName_SYTOXGreen",
    "Image_PathName_SYTOXGreen",
    "Image_FileName_NucleoLive",
    "Image_PathName_NucleoLive",
    "Image_FileName_BF",
    "Image_PathName_BF",
    "Image_ObjectsFileName_Nuclei",
    "Image_ObjectsPathName_Nuclei",
    "Image_ObjectsFileName_Cells",
    "Image_ObjectsPathName_Cells",
]


# In[5]:


# Define paths and regex patterns
raw_pattern = re.compile(
    r"^(?P<Well>[A-Z]\d+_\d+)_T(?P<Time>\d+)_C(?P<Channel>[1-5])_illumcorrect\.tif{1,2}$"
)
mask_pattern = re.compile(
    r"^(?P<Well>[A-Z]\d+_\d+)_T(?P<Time>\d+)_(?P<MaskChannel>cell|nuclei)_mask\.tif{1,2}$"
)

# Get unique well_fov combinations
base_path = image_based_dir / "processed_data"
raw_dir = base_path / "1.illumination_corrected_files"

well_fovs = set()
if raw_dir.exists():
    for well_fov_dir in raw_dir.iterdir():
        if well_fov_dir.is_dir():
            well_fovs.add(well_fov_dir.name)

well_fovs = sorted(list(well_fovs))

# Process each well_fov
for well_fov in tqdm.tqdm(well_fovs):
    raw_image_dir = raw_dir / well_fov
    mask_image_dir = base_path / "2.cell_segmentation_masks" / well_fov

    # Organize images by timepoint
    timepoint_data = {}

    # Collect raw images
    if raw_image_dir.exists():
        for img_file in raw_image_dir.iterdir():
            if img_file.is_file():
                match = raw_pattern.match(img_file.name)
                if match:
                    well = match.group("Well")
                    time = int(match.group("Time"))
                    channel = match.group("Channel")

                    if time not in timepoint_data:
                        timepoint_data[time] = {
                            "well": well,
                            "raw_images": {},
                            "masks": {},
                        }
                    timepoint_data[time]["raw_images"][channel] = img_file

    # Collect mask images
    if mask_image_dir.exists():
        for mask_file in mask_image_dir.iterdir():
            if mask_file.is_file():
                match = mask_pattern.match(mask_file.name)
                if match:
                    time = int(match.group("Time"))
                    mask_channel = match.group("MaskChannel")

                    if time in timepoint_data:
                        timepoint_data[time]["masks"][mask_channel] = mask_file

    written_rows = 0
    incomplete_rows = 0

    for time in sorted(timepoint_data.keys()):
        data = timepoint_data[time]

        # Require all 5 channels + both masks
        has_all_channels = all(str(i) in data["raw_images"] for i in range(1, 6))
        has_both_masks = "nuclei" in data["masks"] and "cell" in data["masks"]

        if not (has_all_channels and has_both_masks):
            incomplete_rows += 1
            continue

        row = {
            "Metadata_Well": data["well"],
            "Metadata_Time": time,
            "Metadata_WellFOV": well_fov,
            "Image_FileName_CL640": data["raw_images"]["1"].name,
            "Image_PathName_CL640": str(data["raw_images"]["1"].parent),
            "Image_FileName_CL488": data["raw_images"]["2"].name,
            "Image_PathName_CL488": str(data["raw_images"]["2"].parent),
            "Image_FileName_SYTOXGreen": data["raw_images"]["3"].name,
            "Image_PathName_SYTOXGreen": str(data["raw_images"]["3"].parent),
            "Image_FileName_NucleoLive": data["raw_images"]["4"].name,
            "Image_PathName_NucleoLive": str(data["raw_images"]["4"].parent),
            "Image_FileName_BF": data["raw_images"]["5"].name,
            "Image_PathName_BF": str(data["raw_images"]["5"].parent),
            "Image_ObjectsFileName_Nuclei": data["masks"]["nuclei"].name,
            "Image_ObjectsPathName_Nuclei": str(data["masks"]["nuclei"].parent),
            "Image_ObjectsFileName_Cells": data["masks"]["cell"].name,
            "Image_ObjectsPathName_Cells": str(data["masks"]["cell"].parent),
        }

        # One file per timepoint, one row per file
        # Write one CSV per timepoint (single-row files)
        load_data_path = pathlib.Path(
            f"{root_dir}/Wave2_data/5.feature_extraction/loadfiles/{well_fov}_T{time:04d}/load_file.csv"
        )
        load_data_path.parent.mkdir(parents=True, exist_ok=True)
        with open(load_data_path, "w", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerow(row)
        written_rows += 1


print("\nAll per-timepoint load files generated successfully!")
