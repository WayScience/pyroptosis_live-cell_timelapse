#!/usr/bin/env python
# coding: utf-8

# In[1]:


import argparse
import logging
import os
import pathlib
import pprint

# Configure logging FIRST, before any other imports
logging.basicConfig(
    filename="tracking.log",
    level=logging.WARNING,  # Changed from DEBUG to WARNING
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
# Suppress ultrack logging completely
logging.getLogger("ultrack").setLevel(logging.ERROR)
logging.getLogger("ultrack.utils").setLevel(logging.ERROR)
logging.getLogger("ultrack.utils.cuda").setLevel(logging.ERROR)
logging.getLogger("ultrack.utils.edge").setLevel(logging.ERROR)

import json

import matplotlib.pyplot as plt
import natsort
import numpy as np
import optuna
import pandas as pd
import seaborn as sns
import tifffile
import torch
from PIL import Image
from rich.pretty import pprint
from ultrack import (
    MainConfig,
    link,
    segment,
    solve,
    to_tracks_layer,
    track,
    tracks_to_zarr,
)
from ultrack.config import MainConfig
from ultrack.imgproc import detect_foreground, robust_invert
from ultrack.tracks import close_tracks_gaps
from ultrack.utils import estimate_parameters_from_labels, labels_to_contours
from ultrack.utils.array import array_apply, create_zarr

os.environ["OMP_NUM_THREADS"] = "8"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import logging

logging.getLogger("ultrack").setLevel(logging.INFO)

from timelapse_utils.file_utils.notebook_init_utils import (
    bandicoot_check,
    init_notebook,
)

root_dir, in_notebook = init_notebook()
if in_notebook:
    import tqdm.notebook as tqdm
else:
    import tqdm


# In[2]:


# def clear_gpu_memory():
#     """
#     A little function to clear gpu memory
#     """

#     torch.cuda.empty_cache()


# clear_gpu_memory()

# begin time, CPU memory peak usage and GPU memory peak usage tracking
import time
import tracemalloc

# import pynvml

# pynvml.nvmlInit()

start_time = time.time()
tracemalloc.start()
# handle = pynvml.nvmlDeviceGetHandleByIndex(0)  # Assuming you want to track GPU 0


# In[3]:


if not in_notebook:
    print("Running as script")
    # set up arg parser
    parser = argparse.ArgumentParser(description="Segment the nuclei of a tiff image")

    parser.add_argument(
        "--well_fov",
        type=str,
        help="Path to the input directory containing the tiff images",
    )
    parser.add_argument(
        "--generate_gif",
        action="store_true",
        help="Whether to generate a gif of the tracks",
    )

    parser.add_argument(
        "--plate_name",
        type=str,
        help="Name of the plate to process (e.g., 'Wave2')",
    )

    args = parser.parse_args()
    well_fov = args.well_fov
    generate_gif = args.generate_gif
    plate_name = args.plate_name
    if generate_gif:
        print("GIF generation is enabled, this may take a while...")

else:
    print("Running in a notebook")
    well_fov = "B2_1"  # example well_fov
    generate_gif = True
    plate_name = "plate_2"  # example plate name

image_base_dir = bandicoot_check(
    root_dir=root_dir,
    bandicoot_mount_path=pathlib.Path(os.path.expanduser("~/mnt/bandicoot")).resolve(),
)

raw_image_input_dir = pathlib.Path(
    image_base_dir / "processed_data" / "0.renamed_files" / plate_name / well_fov
).resolve(strict=True)

segmentation_mask_input_dir = pathlib.Path(
    image_base_dir
    / "processed_data"
    / "2.cell_segmentation_masks"
    / plate_name
    / well_fov
).resolve()

config_output_path = "../results/ultrack_config.json"
with open(config_output_path, "r") as f:
    config_dict = json.load(f)
config = MainConfig.parse_obj(config_dict)

temporary_output_dir = pathlib.Path("../tmp_output").resolve()
figures_output_dir = pathlib.Path("../figures").resolve()
results_output_dir = pathlib.Path("../results").resolve()
temporary_output_dir.mkdir(exist_ok=True)
figures_output_dir.mkdir(exist_ok=True)
results_output_dir.mkdir(exist_ok=True)
config = MainConfig()
pprint(config)


# In[4]:


file_extensions = {".tif", ".tiff"}
# get all the raw image files
raw_images = list(raw_image_input_dir.glob("*"))
raw_images = [f for f in raw_images if f.suffix in file_extensions]
raw_images = sorted(raw_images)

# get all the segmentation mask files
segmentation_masks = list(segmentation_mask_input_dir.glob("*"))
segmentation_masks = [f for f in segmentation_masks if f.suffix in file_extensions]
segmentation_masks = sorted(segmentation_masks)


nuclei_files = [f for f in raw_images if "C4" in f.name.split("_")[3]]
mask_files = [f for f in segmentation_masks if "nuclei" in f.name]

nuclei_files = natsort.natsorted(nuclei_files)
mask_files = natsort.natsorted(mask_files)

print(f"Found {len(mask_files)} segmentation mask files in the input directory")
print(f"Found {len(nuclei_files)} nuclei files in the input directory")


# In[5]:


# read in the masks and create labels
masks = []
for tiff_file in mask_files[:5]:
    img = tifffile.imread(tiff_file)
    masks.append(img)

masks = np.array(masks)

nuclei = []
for tiff_file in nuclei_files[:5]:
    img = tifffile.imread(tiff_file)
    nuclei.append(img)
nuclei = np.array(nuclei)


# In[6]:


image_dims = nuclei[0].shape


# In[7]:


detections = np.zeros((len(masks), image_dims[0], image_dims[1]), dtype=np.uint16)
edges = np.zeros((len(masks), image_dims[0], image_dims[1]), dtype=np.uint16)
for frame_index, frame in tqdm.tqdm(
    enumerate(masks), total=len(masks), desc="Processing frames"
):
    detections[frame_index, :, :], edges[frame_index, :, :] = labels_to_contours(frame)
print(detections.shape, edges.shape)

# clear_gpu_memory()


# In[8]:


params_df = estimate_parameters_from_labels(masks, is_timelapse=True)
if in_notebook:
    params_df["area"].plot(kind="hist", bins=100, title="Nuclei Area histogram")


# In[9]:


config.segmentation_config.min_area = 50
config.segmentation_config.max_area = 5000
config.segmentation_config.n_workers = 1
config.segmentation_config.threshold = 0.25


config.linking_config.max_distance = 25

config.tracking_config.appear_weight = -1
config.tracking_config.disappear_weight = -1
config.tracking_config.division_weight = -0.1
config.tracking_config.power = 4
config.tracking_config.bias = -0.001
config.tracking_config.solution_gap = 0.0


# In[10]:


track(
    foreground=detections,
    edges=edges,
    config=config,
    overwrite=True,
)


# In[11]:


tracks_df, graph = to_tracks_layer(config)
# tracks_df = close_tracks_gaps(
#     tracks_df=tracks_df,
#     max_gap=2,
#     max_radius=50,
#     spatial_columns=["y", "x"],
# )


# In[12]:


labels = tracks_to_zarr(config, tracks_df)
# save the tracks as parquet
tracks_df.to_parquet(
    f"{results_output_dir}/{well_fov}_tracks.parquet",
)
print(tracks_df["track_id"].nunique())
print(f"Found {tracks_df['track_id'].nunique()} unique tracks in the dataset.")
tracks_df.head()


# In[13]:


if in_notebook:
    # get the number of unique objects per frame and the unique tracks per frame
    unique_objects_per_frame_dict = {
        "frame": [],
        "n_unique_objects": [],
    }
    for frame_number, frame_image in enumerate(masks):
        unique_objects_per_frame_dict["frame"].append(frame_number)
        unique_objects_per_frame_dict["n_unique_objects"].append(
            np.unique(frame_image).size
        )
    unique_objects_per_frame_df = pd.DataFrame(unique_objects_per_frame_dict)

    # plot the number of unique objects per frame and the unique tracks per frame
    objects_per_frame = tracks_df.groupby("t")["track_id"].nunique()
    plt.figure(figsize=(10, 6))
    # plot the first line (unique objects per frame)
    plt.plot(
        unique_objects_per_frame_df["frame"],
        unique_objects_per_frame_df["n_unique_objects"],
        label="Unique Objects",
    )
    # plot the second line (unique tracks per frame)
    plt.plot(objects_per_frame.index, objects_per_frame.values, label="Unique Tracks")
    plt.title("Number of Unique Objects and Tracks per Frame")
    plt.xlabel("Frame")
    plt.ylabel("Number of Unique Items")
    plt.legend()
    plt.savefig(
        f"{figures_output_dir}/{well_fov}_unique_objects_and_tracks_per_frame.png"
    )
    plt.show()


# In[14]:


if generate_gif:
    tracks_df.reset_index(drop=True, inplace=True)
    cum_tracks_df = tracks_df.copy()
    # zero out the track_df for plotting
    cum_tracks_df = cum_tracks_df.loc[cum_tracks_df["t"] == -1]
    for frame_index, _ in enumerate(nuclei):
        tmp_df = tracks_df.loc[tracks_df["t"] == frame_index]
        cum_tracks_df = pd.concat([cum_tracks_df, tmp_df])
        plt.figure(figsize=(6, 3))
        plt.subplot(121)
        plt.imshow(detections[frame_index, :, :], cmap="gray")
        plt.title("Detections")
        plt.axis("off")

        plt.subplot(122)
        sns.lineplot(
            data=cum_tracks_df,
            x="x",
            y="y",
            hue="track_id",
            legend=False,
            palette="Spectral",
            linewidth=0.8,
            alpha=0.8,
        )
        plt.imshow(detections[frame_index, :, :], cmap="gray", alpha=0.3)
        plt.title(f"Frame {frame_index}")
        plt.axis("off")

        plt.tight_layout()
        plt.savefig(f"{temporary_output_dir}/tracks_{frame_index}.png")
        plt.close()
    # load each image
    files = [f for f in temporary_output_dir.glob("*.png")]
    files = sorted(files, key=lambda x: int(x.stem.split("_")[1]))
    frames = [Image.open(f) for f in files]
    fig_path = figures_output_dir / f"{well_fov}_tracks.gif"
    # plot the line of each track in matplotlib over a gif
    # get the tracks
    # save the frames as a gif
    frames[0].save(
        fig_path, save_all=True, append_images=frames[1:], duration=750, loop=0
    )


# In[15]:


# # clean up tracking files
# # remove temporary_output_dir
# shutil.rmtree(temporary_output_dir)

# track_db_path = pathlib.Path("data.db").resolve()
# metadata_toml_path = pathlib.Path("metadata.toml").resolve()
# if track_db_path.exists():
#     track_db_path.unlink()
# if metadata_toml_path.exists():
#     metadata_toml_path.unlink()


# In[16]:


# clear_gpu_memory()


# In[17]:


end_time = time.time()
current, peak = tracemalloc.get_traced_memory()
# mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
print(f"Time taken for tracking: {end_time - start_time:.2f} seconds")
print(f"Peak memory used: {peak / (1024**3):.2f} GB")
# print(f"GPU Memory used: {mem_info.used / (1024**3):.2f} GB")
