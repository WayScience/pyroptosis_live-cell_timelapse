#!/usr/bin/env python
# coding: utf-8

# # Run CellProfiler `illum.cppipe` (IC) pipeline
#
# In this notebook, we run the CellProfiler IC pipeline to calculate the illumination (illum) correction functions for all images per channel (5), apply the functions, and save images into a new directory.

# ## Import libraries

# In[1]:


import argparse
import os
import pathlib
import time

from timelapse_utils.cp_utils.cp_utils import run_cellprofiler
from timelapse_utils.file_utils.notebook_init_utils import (
    bandicoot_check,
    init_notebook,
)

root_dir, in_notebook = init_notebook()


# ## Set paths

# In[2]:


if not in_notebook:
    print("Running as script")
    # set up arg parser
    parser = argparse.ArgumentParser(description="Illumination correction")

    parser.add_argument(
        "--well_fov",
        type=str,
        help="Path to the input directory containing the tiff images",
    )

    args = parser.parse_args()
    well_fov = args.well_fov
else:
    print("Running in a notebook")
    well_fov = "B2_1"

image_base_dir = bandicoot_check(
    bandicoot_mount_path=pathlib.Path(
        os.path.expanduser(
            "~/mnt/bandicoot/live_cell_timelapse_pyroptosis_project_data/processed_data"
        )
    ).resolve(),
    root_dir=root_dir,
)
image_base_dir = pathlib.Path(
    f"{image_base_dir}/live_cell_timelapse_pyroptosis_project_data/processed_data"
).resolve(strict=True)
run_name = "illumination_correction"
# path to folder for IC images
illum_directory = pathlib.Path(
    f"{image_base_dir}/1.illumination_corrected_files"
).resolve()
# make sure the directory exists
illum_directory.mkdir(exist_ok=True, parents=True)
input_dir = pathlib.Path(f"{image_base_dir}/0.renamed_files/{well_fov}").resolve(
    strict=True
)


# ## Define the input paths

# 5 FOVs per well, 96 wells per plate, 1 plate at 18 time points = 8640 image sets

# In[3]:


path_to_pipeline = pathlib.Path("../pipelines/illum_5ch.cppipe").resolve(strict=True)
# get all directories with raw images


dict_of_runs = {}
dict_of_runs[input_dir.stem] = {
    "path_to_images": str(input_dir),
    "path_to_output": str(illum_directory / input_dir.stem),
    "path_to_pipeline": path_to_pipeline,
}
print(f"Added {len(dict_of_runs.keys())} to the list of runs")
print(f"Running {input_dir.stem}")


# ## Run `illum.cppipe` pipeline and calculate + save IC images
# This last cell does not get run as we run this pipeline in the command line.

# In[4]:


start = time.time()


# In[5]:


run_cellprofiler(
    path_to_pipeline=dict_of_runs[input_dir.stem]["path_to_pipeline"],
    path_to_input=dict_of_runs[input_dir.stem]["path_to_images"],
    path_to_output=dict_of_runs[input_dir.stem]["path_to_output"],
)


# In[6]:


end = time.time()
# format the time taken into hours, minutes, seconds
hours, rem = divmod(end - start, 3600)
minutes, seconds = divmod(rem, 60)
print(
    "Total time taken: {:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds)
)
