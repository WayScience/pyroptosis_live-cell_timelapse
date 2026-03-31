#!/usr/bin/env python
# coding: utf-8

# # Annotate merged single cells with metadata from platemap file

# ## Import libraries

# In[1]:


import os
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pycytominer import annotate
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


# ## Set paths and variables
# ### Relate the CellProfiler output to the platemap file

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
qc_profiles_path = pathlib.Path(
    f"{image_base_dir}/6.qc_profiles/qc_profiles.parquet"
).resolve(strict=True)

annotated_profiles_path = pathlib.Path(
    f"{image_base_dir}/7.annotated_profiles/annotated_profiles.parquet"
).resolve()
annotated_profiles_path.parent.mkdir(exist_ok=True)


# ## Annotate merged single cells

# In[3]:


# load in converted parquet file as df to use in annotate function
single_cell_df = pd.read_parquet(qc_profiles_path)

platemap_df = pd.read_csv(platemap_path)


# add metadata from platemap file to extracted single cell features
annotated_df = annotate(
    profiles=single_cell_df,
    platemap=platemap_df,
    join_on=["Metadata_Well", "Metadata_Well"],
)


# In[4]:


# save annotated df as parquet file
output(
    df=annotated_df,
    output_filename=annotated_profiles_path,
    output_type="parquet",
)

annotated_df.head()
