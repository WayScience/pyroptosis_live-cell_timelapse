#!/usr/bin/env python
# coding: utf-8

# This notebook focuses on trying to find a way to segment nuclei properly.
# The end goals is to segment cell and extract morphology features from cellprofiler.
# These masks must be imported into cellprofiler to extract features.

# In[1]:


import argparse
import os
import pathlib

import matplotlib.pyplot as plt
import natsort

# Import dependencies
import numpy as np
import skimage
import tifffile
import torch
from cellpose import models
from csbdeep.utils import normalize
from PIL import Image
from timelapse_utils.file_utils.notebook_init_utils import (
    bandicoot_check,
    init_notebook,
)

root_dir, in_notebook = init_notebook()
if in_notebook:
    import tqdm.notebook as tqdm
else:
    import tqdm


# In[ ]:


if not in_notebook:
    # set up arg parser
    parser = argparse.ArgumentParser(description="Segment the nuclei of a tiff image")

    parser.add_argument(
        "--well_fov",
        type=str,
        help="Name of the well and field of view to segment, e.g. B2_1",
    )

    parser.add_argument(
        "--clip_limit",
        type=float,
        help="Clip limit for the adaptive histogram equalization",
    )

    args = parser.parse_args()
    clip_limit = args.clip_limit
    well_fov = pathlib.Path(args.well_fov).resolve(strict=True)


else:
    well_fov = "B2_2"
    clip_limit = 0.6


image_base_dir = bandicoot_check(
    root_dir=root_dir,
    bandicoot_mount_path=pathlib.Path(os.path.expanduser("~/mnt/bandicoot")).resolve(),
)

input_dir = pathlib.Path(
    image_base_dir
    / "live_cell_timelapse_pyroptosis_project_data"
    / "processed_data"
    / "1.illumination_corrected_files"
    / well_fov
).resolve(strict=True)

segmentation_mask_output_dir = pathlib.Path(
    image_base_dir
    / "live_cell_timelapse_pyroptosis_project_data"
    / "processed_data"
    / "2.cell_segmentation_masks"
    / well_fov
).resolve()
segmentation_mask_output_dir.mkdir(exist_ok=True, parents=True)


figures_dir = pathlib.Path("../figures").resolve()
figures_dir.mkdir(exist_ok=True, parents=True)

if optimize_segmentation:
    print("Optimizing Segmentation")
elif not optimize_segmentation:
    print("Running segmentation")


# In[3]:


# set up memory profiler for GPU
device = torch.device("cuda:0")
free_before, total_before = torch.cuda.mem_get_info(device)
starting_level_GPU_RAM = (total_before - free_before) / 1024**2
print("Starting level of GPU RAM available (MB): ", starting_level_GPU_RAM)


# ## Set up images, paths and functions

# In[4]:


image_extensions = {".tif", ".tiff"}
files = sorted(input_dir.glob("*"))
files = [str(x) for x in files if x.suffix in image_extensions]
files = natsort.natsorted(files)


# In[5]:


image_dict = {
    "nuclei_file_paths": [],
    "nuclei": [],
}


# In[6]:


# split files by channel
for file in files:
    if "C4" in file.split("/")[-1]:
        image_dict["nuclei_file_paths"].append(file)
        image_dict["nuclei"].append(tifffile.imread(file).astype(np.float32))
nuclei_image_list = [np.array(nuclei) for nuclei in image_dict["nuclei"]]

nuclei = np.array(nuclei_image_list).astype(np.int16)

nuclei = skimage.exposure.equalize_adapthist(nuclei, clip_limit=clip_limit)

print(nuclei.shape)


# In[7]:


original_nuclei_image = nuclei.copy()


# ## Cellpose

# ### Runnning segmentation

# In[8]:


use_GPU = torch.cuda.is_available()
model = models.CellposeModel(
    gpu=use_GPU,
)
masks_all_dict = {"masks": [], "imgs": []}

# get masks for all the images
# save to a dict for later use
for img in tqdm.tqdm(nuclei, desc="Segmenting nuclei"):
    img = normalize(img)
    masks, flows, styles = model.eval(img)

    masks_all_dict["masks"].append(masks)
    masks_all_dict["imgs"].append(img)
print(len(masks_all_dict))
masks_all = masks_all_dict["masks"]
imgs = masks_all_dict["imgs"]

masks_all = np.array(masks_all)
imgs = np.array(imgs)


# In[9]:


for frame_index, frame in enumerate(image_dict["nuclei_file_paths"]):
    # saving the masks
    save_file_path = f"{segmentation_mask_output_dir}/{str(frame).split('/')[-1].split('_C4')[0]}_nuclei_mask.tiff"
    tifffile.imwrite(save_file_path, masks_all[frame_index, :, :])
if in_notebook:
    for z in range(len(masks_all)):
        plt.figure(figsize=(20, 10))

        plt.title(f"z: {z}")
        plt.axis("off")
        plt.subplot(1, 2, 1)
        plt.imshow(nuclei[z], cmap="inferno")
        plt.title("Nuclei")
        plt.axis("off")

        plt.subplot(122)
        plt.imshow(masks_all[z], cmap="nipy_spectral")
        plt.title("Cell masks")
        plt.axis("off")
        plt.show()


# In[10]:


# set up memory profiler for GPU
device = torch.device("cuda:0")
free_after, total_after = torch.cuda.mem_get_info(device)
amount_used = ((total_after - free_after)) / 1024**2
print(f"Used: {amount_used} MB or {amount_used / 1024} GB of GPU RAM")
