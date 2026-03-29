#!/usr/bin/env python
# coding: utf-8

# # Perform segmentation and feature extraction for each plate using CellProfiler Parallel

# ## Import libraries

# In[1]:


import argparse
import os
import pathlib
import shutil
import time

from timelapse_utils.cp_utils.cp_parallel import run_cellprofiler_parallel
from timelapse_utils.file_utils.notebook_init_utils import (
    bandicoot_check,
    init_notebook,
)

root_dir, in_notebook = init_notebook()


image_based_dir = bandicoot_check(
    bandicoot_mount_path=pathlib.Path(f"{os.path.expanduser('~')}/mnt/bandicoot/"),
    root_dir=root_dir,
)
image_based_dir = image_based_dir / "live_cell_timelapse_pyroptosis_project_data"


# In[2]:


if in_notebook:
    import tqdm.notebook as tqdm

    max_workers = 10
else:
    import tqdm

    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "--max_workers",
        type=int,
        default=None,
        help="The maximum number of workers to use for parallel processing. If not specified, the number of workers will be set to the number of CPU cores minus 2.",
    )
    args = argparser.parse_args()
    max_workers = args.max_workers


# ## Set paths and variables

# In[3]:


path_to_pipeline = pathlib.Path(
    f"{root_dir}/Wave2_data/5.feature_extraction/pipelines/analysis_5ch.cppipe"
).resolve(strict=True)
load_file_dir = pathlib.Path(f"{root_dir}/Wave2_data/5.feature_extraction/loadfiles/")


# In[ ]:


# find all dirs in loadfiles path that contain the well_fov name (one per timepoint)
timepoint_dirs = sorted(load_file_dir.glob(f"*/"))[:1000]


# ## Create dictionary with all info for each well

# In[5]:


# get all directories with raw images
dict_of_runs = {}
for timepoint_dir in tqdm.tqdm(timepoint_dirs):
    dict_of_runs[timepoint_dir.name] = {
        "path_to_images": str(timepoint_dir),
        "path_to_output": str(
            pathlib.Path(
                f"{root_dir}/Wave2_data/5.feature_extraction/extracted_features/{timepoint_dir.name}"
            ).resolve()
        ),
        "path_to_final_output": str(
            pathlib.Path(
                f"{image_based_dir}/processed_data/3.extracted_features/{timepoint_dir.name}"
            ).resolve()
        ),
        "path_to_pipeline": path_to_pipeline,
    }
    pathlib.Path(dict_of_runs[timepoint_dir.name]["path_to_output"]).mkdir(
        exist_ok=True, parents=True
    )
    pathlib.Path(dict_of_runs[timepoint_dir.name]["path_to_final_output"]).mkdir(
        exist_ok=True, parents=True
    )
    # check if there is a file in the final output dir
    # if so then remove this timepoint from the dict of runs
    if (
        len(
            list(
                pathlib.Path(
                    dict_of_runs[timepoint_dir.name]["path_to_final_output"]
                ).glob("*")
            )
        )
        > 0
    ):
        # remove this record from the run dict
        dict_of_runs.pop(timepoint_dir.name, None)
dict_of_runs


# ## Run analysis pipeline on each plate in parallel
#
# This cell is not finished to completion due to how long it would take. It is ran in the python file instead.

# In[6]:


start = time.time()


# In[7]:


run_cellprofiler_parallel(
    plate_info_dictionary=dict_of_runs,
    run_name="pyroptosis_live_cell_timelapse_analysis",
    log_dir=pathlib.Path(f"{root_dir}/Wave2_data/5.feature_extraction/logs/").resolve(),
    max_workers=max_workers,  # adjust this based on your system's capabilities
)


# In[8]:


end = time.time()
# format the time taken into hours, minutes, seconds
hours, rem = divmod(end - start, 3600)
minutes, seconds = divmod(rem, 60)
print(
    "Total time taken: {:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds)
)


# In[9]:


# loop through the dict of runs and move the output files to the final output directory
# thie final output file is on a NAS and cellprofiler cannot update the write file in place, so we need to move the file to the final output directory
for well_fov_timepoint in dict_of_runs.keys():
    tmp_output_file_path = pathlib.Path(
        f"{dict_of_runs[well_fov_timepoint]['path_to_output']}/pyroptosis_timelapse.sqlite"
    )
    final_output_file_path = pathlib.Path(
        f"{dict_of_runs[well_fov_timepoint]['path_to_final_output']}/{well_fov_timepoint}.sqlite"
    )
    if not tmp_output_file_path.exists():
        continue
    final_output_file_path.parent.mkdir(parents=True, exist_ok=True)
    if final_output_file_path.exists():
        final_output_file_path.unlink()
    # use move (copy+remove fallback) to support cross-device paths
    shutil.move(str(tmp_output_file_path), str(final_output_file_path))
