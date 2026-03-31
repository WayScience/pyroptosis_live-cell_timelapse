#!/usr/bin/env python
# coding: utf-8

# # Merge single cells from CellProfiler outputs using CytoTable

# In[1]:


import os
import pathlib
import time
import uuid

import natsort
import pandas as pd
from cytotable import convert, presets
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

from parsl.config import Config
from parsl.executors import HighThroughputExecutor

# ## set config joins for each preset

# In[2]:


# preset configurations based on typical CellProfiler outputs
preset = "cellprofiler_sqlite_pycytominer"
presets.config[preset][
    "CONFIG_JOINS"
    # remove Image_Metadata_Plate from SELECT as this metadata was not extracted from file names
    # add Image_Metadata_FOV as this is an important metadata when finding where single cells are located
] = """WITH Per_Image_Filtered AS (
                SELECT
                    Metadata_ImageNumber,
                    Image_URL_CL488,
                    Image_URL_CL640,
                    Image_URL_NucleoLive,
                    Image_URL_BF,
                    Image_URL_SYTOXGreen,
                FROM
                    read_parquet('per_image.parquet')
                )
            SELECT
                *
            FROM
                Per_Image_Filtered AS per_image
            LEFT JOIN read_parquet('per_cytoplasm.parquet') AS per_cytoplasm ON
                per_cytoplasm.Metadata_ImageNumber = per_image.Metadata_ImageNumber
            LEFT JOIN read_parquet('per_cells.parquet') AS per_cells ON
                per_cells.Metadata_ImageNumber = per_cytoplasm.Metadata_ImageNumber
                AND per_cells.Metadata_Cells_Number_Object_Number = per_cytoplasm.Metadata_Cytoplasm_Parent_Cells
            LEFT JOIN read_parquet('per_nuclei.parquet') AS per_nuclei ON
                per_nuclei.Metadata_ImageNumber = per_cytoplasm.Metadata_ImageNumber
                AND per_nuclei.Metadata_Nuclei_Number_Object_Number = per_cytoplasm.Metadata_Cytoplasm_Parent_Nuclei
            """


# ## Set paths and variables
#
# All paths must be string but we use pathlib to show which variables are paths

# In[3]:


image_base_dir = bandicoot_check(
    bandicoot_mount_path=pathlib.Path(f"{os.path.expanduser('~')}/mnt/bandicoot/"),
    root_dir=root_dir,
)
image_base_dir = pathlib.Path(
    f"{image_base_dir}/live_cell_timelapse_pyroptosis_project_data/processed_data/"
).resolve(strict=True)
extracted_features_dir = pathlib.Path(
    f"{image_base_dir}/3.extracted_features/"
).resolve(strict=True)


# In[4]:


# type of file output from CytoTable (currently only parquet)
dest_datatype = "parquet"

# directory where parquet files are saved to
output_dir = pathlib.Path(f"{image_base_dir}/4.converted_profiles/").resolve()
output_dir.mkdir(exist_ok=True, parents=True)


# ## Gather all sqlite files for each well_fov_timepoint

# In[5]:


# well_fov_timepoints
well_fov_timepoints = [
    x for x in tqdm.tqdm(extracted_features_dir.glob("*")) if x.is_dir()
]
well_fov_timepoints = natsort.natsorted(well_fov_timepoints)[:10]
well_fov_timepoints_sqlites = [
    list(x.glob("**/*.sqlite"))[0]
    for x in tqdm.tqdm(well_fov_timepoints)
    if len(list(x.glob("**/*.sqlite"))) > 0
]
well_fov_timepoints_sqlites = natsort.natsorted(well_fov_timepoints_sqlites)


# ## Convert SQLite file and merge single cell objects into parquet file
#
# This was not run to completion as we use the nbconverted python file for full run.

# In[6]:


times_dict = {
    "well_fov_timepoint": [],
    "time": [],
}
for well_fov_timepoint_sqlite_file_path in tqdm.tqdm(well_fov_timepoints_sqlites):
    dest_path = output_dir / str(well_fov_timepoint_sqlite_file_path.stem)
    dest_path.mkdir(exist_ok=True, parents=True)
    dest_path = (
        dest_path / f"{well_fov_timepoint_sqlite_file_path.stem}.{dest_datatype}"
    )
    if dest_path.exists():
        print(f"{dest_path} already exists, skipping...")
        continue
    # extract key metadata that CP did not capture
    # this is at the file level
    # well, fov, timepoint are all captured in the file name so we can extract them from there
    well_fov_timepoint = well_fov_timepoint_sqlite_file_path.stem
    well = well_fov_timepoint.split("_")[0]
    fov = well_fov_timepoint.split("_")[1]
    timepoint = well_fov_timepoint.split("_")[2].strip("T")
    timepoint = int(timepoint)
    # set up the time profiling
    start_time = time.time()
    # merge single cells and output as parquet file
    convert(
        source_path=well_fov_timepoint_sqlite_file_path,
        dest_path=dest_path,
        dest_datatype=dest_datatype,
        preset=preset,
        parsl_config=Config(
            executors=[HighThroughputExecutor()],
            run_dir=f"cytotable_runinfo/{uuid.uuid4().hex}",
        ),
        chunk_size=1000,
    )
    # read in the parquet ouput from cytotable and clean the df a lil bit

    df = pd.read_parquet(dest_path)
    # set columns to drop
    df.drop(
        columns=[
            "Metadata_ImageNumber_1",
            "Metadata_ImageNumber_2",
            "Metadata_ImageNumber_3",
        ],
        inplace=True,
    )
    # establish which columns are metadata columns to move to Metadata_ prefix

    location_metadata = [
        x
        for x in df.columns
        if "center_" in x.lower()
        or "location" in x.lower()
        or "boundingbox" in x.lower()
    ]
    # set columns to move to Metadata_ prefix
    file_metadata = [
        "Image_URL_BF",
        "Image_URL_CL488",
        "Image_URL_CL640",
        "Image_URL_NucleoLive",
        "Image_URL_SYTOXGreen",
    ]
    object_metadata = [
        x
        for x in df.columns
        if "count" in x.lower()
        or ("parent" in x.lower() and "metadata" not in x.lower())
    ]
    # rename columns to have Metadata_ prefix for metadata columns
    df.rename(
        columns={
            x: f"Metadata_{x}"
            for x in file_metadata + object_metadata + location_metadata
            if "Metadata" not in x
        },
        inplace=True,
    )
    df.insert(0, "Metadata_Well", f"{well}")
    df.insert(1, "Metadata_Time", f"{timepoint}")
    df.insert(2, "Metadata_FOV", f"{fov}")
    df.insert(2, "Metadata_Well_FOV", df["Metadata_Well"] + "_" + df["Metadata_FOV"])
    df.insert(
        4, "Metadata_Well_FOV_Time", df["Metadata_Well_FOV"] + "_" + df["Metadata_Time"]
    )
    df["Metadata_single_cell_count"] = df.shape[0]
    df.to_parquet(dest_path)
    end_time = time.time()
    times_dict["time"].append(end_time - start_time)
    times_dict["well_fov_timepoint"].append(well_fov_timepoint)


# In[7]:


df = pd.DataFrame(times_dict)
df.head()
