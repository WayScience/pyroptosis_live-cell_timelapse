#!/usr/bin/env python
# coding: utf-8

# # Perform segmentation and feature extraction for each plate using CellProfiler Parallel

# ## Import libraries

# In[1]:


import argparse
import pathlib
import sys
import time

sys.path.append("../../../utils/")
import cp_utils

# check if in a jupyter notebook
try:
    cfg = get_ipython().config
    in_notebook = True
except NameError:
    in_notebook = False


# ## Set paths and variables

# In[2]:


if not in_notebook:
    print("Running as script")
    # set up arg parser
    parser = argparse.ArgumentParser(description="Illumination correction")

    parser.add_argument(
        "--input_dir",
        type=str,
        help="Path to the input directory containing the tiff images",
    )

    args = parser.parse_args()
    images_dir = pathlib.Path(args.input_dir).resolve(strict=True)
else:
    print("Running in a notebook")
    images_dir = pathlib.Path(
        "../../2.illumination_correction/illum_directory/W0052_F0001/"
    )

# path to plugins directory as one of the pipelines uses the RunCellpose plugin
plugins_dir = pathlib.Path(
    "/home/lippincm/Documents/CellProfiler-plugins/active_plugins"
)
path_to_pipeline = pathlib.Path("../pipelines/analysis_5ch.cppipe").resolve(strict=True)

# set main output dir for all plates
output_dir = pathlib.Path(f"../analysis_output/{images_dir.stem}")
output_dir.mkdir(exist_ok=True, parents=True)


# ## Create dictionary with all info for each well

# In[3]:


# get all directories with raw images
dict_of_runs = {}

dict_of_runs[images_dir.stem] = {
    "path_to_images": str(images_dir),
    "path_to_output": str(output_dir),
    "path_to_pipeline": path_to_pipeline,
}
print(f"Running {images_dir.stem}")


# ## Run analysis pipeline on each plate in parallel
# 
# This cell is not finished to completion due to how long it would take. It is ran in the python file instead.

# In[4]:


start = time.time()


# In[5]:


cp_utils.run_cellprofiler(
    path_to_pipeline=dict_of_runs[images_dir.stem]["path_to_pipeline"],
    path_to_input=dict_of_runs[images_dir.stem]["path_to_images"],
    path_to_output=dict_of_runs[images_dir.stem]["path_to_output"],
)


# In[6]:


end = time.time()
# format the time taken into hours, minutes, seconds
hours, rem = divmod(end - start, 3600)
minutes, seconds = divmod(rem, 60)
print(
    "Total time taken: {:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds)
)

