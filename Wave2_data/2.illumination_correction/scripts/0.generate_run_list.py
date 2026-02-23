#!/usr/bin/env python
# coding: utf-8

# In[6]:


import gc
import os
import pathlib
import re

import pandas as pd
import tqdm
from timelapse_utils.file_utils.notebook_init_utils import (
    bandicoot_check,
    init_notebook,
)
from timelapse_utils.image_utils.timelapse_image_utils import natural_key

root_dir, in_notebook = init_notebook()

if in_notebook:
    import tqdm.notebook as tqdm
else:
    import tqdm


# In[7]:


bandicoot_mount_path = pathlib.Path(os.path.expanduser("~/mnt/bandicoot/"))
image_base_dir = bandicoot_check(bandicoot_mount_path, root_dir)


# In[8]:


# repository data directory to access the data faster
path_to_processed_data = pathlib.Path(
    f"{image_base_dir}/live_cell_timelapse_pyroptosis_project_data/processed_data/0.renamed_files/"
).resolve()

# save path
path_to_corrected_images = pathlib.Path(
    f"{image_base_dir}/live_cell_timelapse_pyroptosis_project_data/processed_data/1.illumination_corrected_files/"
).resolve()
path_to_corrected_images.mkdir(exist_ok=True, parents=True)


# In[ ]:


# get a list of all well_fov dirs
well_fovs = path_to_processed_data.glob("*")
well_fovs = sorted(well_fovs)
well_fovs = [d for d in well_fovs if d.is_dir()]
# get a list of all .tif or .tiff files in the well_fov dirs
list_of_files = []
for well_fov in well_fovs:
    files = well_fov.glob("*.tif*")
    list_of_files.extend(files)
list_of_files = sorted(list_of_files)
# natural sort the file names
list_of_files = [
    pathlib.Path(str(f))
    for f in sorted(list_of_files, key=lambda x: natural_key(x.name))
]
expected_files_df = pd.DataFrame({"file_path": list_of_files})


# In[17]:


expected_files_df["file_name"] = expected_files_df["file_path"].apply(lambda x: x.name)
expected_files_df["file_parent"] = expected_files_df["file_path"].apply(
    lambda x: x.parent
)
expected_files_df["new_parent"] = expected_files_df["file_parent"].apply(
    lambda x: re.sub(r"0\.renamed_files", "1.illumination_corrected_files", str(x))
)
expected_files_df["new_file_path"] = expected_files_df.apply(
    lambda row: row["new_parent"] + "/" + row["file_name"], axis=1
)
expected_files_df["well_fov"] = expected_files_df["file_parent"].apply(
    lambda x: pathlib.Path(x).name
)

# get the well_fovs to run
well_fovs_to_run = expected_files_df.loc[
    ~expected_files_df["new_file_path_exists"], "well_fov"
].unique()
print(f"Total number of files to process: {len(expected_files_df)}")
print(f"Number of well_fovs to run: {len(well_fovs_to_run)}")


# In[18]:


# save the well_fovs to run to a tsv file
mkdir_path = pathlib.Path("../loadfiles/")
mkdir_path.mkdir(exist_ok=True, parents=True)
pd.DataFrame({"well_fov": well_fovs_to_run}).to_csv(
    "../loadfiles/well_fovs_to_run.tsv", index=False, sep="\t"
)
