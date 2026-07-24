#!/usr/bin/env python
# coding: utf-8

# In[1]:


import argparse
import logging
import os
import pathlib
import time

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
    import napari
    import tqdm

# begin time, CPU memory peak usage and GPU memory peak usage tracking
import time
import tracemalloc

import napari
import pynvml
import torch
import tracksdata as td
from dask.array.image import imread
from hoct import load_model, predict
from hoct.features import normalize_image
from hoct.tracking import ILPSolverConfig

# def clear_gpu_memory():
#     """
#     A little function to clear gpu memory
#     """

#     torch.cuda.empty_cache()


# clear_gpu_memory()


pynvml.nvmlInit()

start_time = time.time()
tracemalloc.start()
handle = pynvml.nvmlDeviceGetHandleByIndex(0)  # Assuming you want to track GPU 0


# In[2]:


start_time = time.time()


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
    generate_gif = False
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

track_save_path = pathlib.Path(
    image_base_dir / "processed_data" / "3.cell_tracks" / plate_name / well_fov
).resolve()


temporary_output_dir = pathlib.Path("../tmp_output").resolve()
figures_output_dir = pathlib.Path("../figures").resolve()
results_output_dir = pathlib.Path("../results").resolve()
track_save_path.mkdir(exist_ok=True, parents=True)
temporary_output_dir.mkdir(exist_ok=True, parents=True)
figures_output_dir.mkdir(exist_ok=True, parents=True)
results_output_dir.mkdir(exist_ok=True, parents=True)


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
labels = []
for tiff_file in mask_files:
    img = tifffile.imread(tiff_file)
    labels.append(img)

labels = np.array(labels)

images = []
for tiff_file in nuclei_files:
    img = tifffile.imread(tiff_file)
    images.append(img)
images = np.array(images)


# In[6]:


# Load the default pre-trained model (downloaded and cached on first use).
device = "cuda" if torch.cuda.is_available() else "cpu"
model = load_model(device=device)

# optionally provide the ilp solver config, it could be None for default config
solver_config = ILPSolverConfig(
    appearance_weight=0.5,
    delta_t_weight=0.5,
    disappearance_weight=0.25,
    division_weight=0.25,
    edge_bias=0.5,
    node_weight=-10.0,
    tracklet_solver=True,
)

# Run tracking
graph = predict(
    model=model,
    labels=labels,
    images=images,
    solver_config=solver_config,
    distance_threshold=300.0,
    n_neighbors=5,
    max_delta_t=3,
    # this is only required for large volumes where tiled prediction is needed
    tiling_scheme=td.functional.TilingScheme(
        tile_shape=(1, 32, 256, 256),
        overlap_shape=(2, 16, 120, 120),
    ),
    test_time_augs=5,  # optional, takes longer but it improves performance
)


# In[7]:


# Visualize
tracks_df, track_graph, track_labels = td.functional.to_napari_format(
    graph, mask_key="mask"
)


# In[8]:


# viewer = napari.Viewer()
# viewer.add_image(images, name="images")
# viewer.add_labels(track_labels, name="track_labels")
# viewer.add_tracks(tracks_df, graph=track_graph, name="tracks")
# viewer.add_labels(labels, name="labels")

# napari.run()


# In[9]:


# convert the DataFrame object to a pandas DataFrame and retain the column names
tracks_df = pd.DataFrame(tracks_df, columns=tracks_df.columns)
# save the tracks
tracks_df.to_parquet(pathlib.Path(track_save_path / "cell_tracks.parquet"), index=False)
tracks_df.head()


# In[10]:


if generate_gif:
    # tracks_df.reset_index(drop=True, inplace=True)
    cum_tracks_df = tracks_df.copy()
    # zero out the track_df for plotting
    cum_tracks_df = cum_tracks_df.loc[cum_tracks_df["t"] == -1]
    for frame_index, _ in enumerate(images):
        tmp_df = tracks_df.loc[tracks_df["t"] == frame_index]
        cum_tracks_df = pd.concat([cum_tracks_df, tmp_df])
        plt.figure(figsize=(6, 3))
        plt.subplot(121)
        plt.imshow(labels[frame_index, :, :], cmap="gray")
        plt.title("Images")
        plt.axis("off")

        plt.subplot(122)
        sns.lineplot(
            data=cum_tracks_df,
            x="x",
            y="y",
            hue="tracklet_id",
            legend=False,
            palette="Spectral",
            linewidth=0.8,
            alpha=0.8,
        )
        plt.imshow(labels[frame_index, :, :], cmap="gray", alpha=0.3)
        plt.title(f"Frame {frame_index}")
        plt.axis("off")

        plt.tight_layout()
        plt.savefig(f"{temporary_output_dir}/tracks_{frame_index}.png")
        plt.close()
    # load each image
    files = [f for f in temporary_output_dir.glob("*.png")]
    files = sorted(files, key=lambda x: int(x.stem.split("_")[1]))
    frames = [Image.open(f) for f in files]
    figures_output_dir = pathlib.Path("../figures")
    figures_output_dir.mkdir(exist_ok=True)
    fig_path = figures_output_dir / f"{well_fov}_tracks.gif"
    # plot the line of each track in matplotlib over a gif
    # get the tracks
    # save the frames as a gif
    frames[0].save(
        fig_path, save_all=True, append_images=frames[1:], duration=750, loop=0
    )


# In[11]:


end_time = time.time()
end_mem = tracemalloc.get_traced_memory()[1]
end_gpu_mem = pynvml.nvmlDeviceGetMemoryInfo(handle).used
print(f"Total time: {end_time - start_time:.2f} seconds")
print(f"Peak CPU memory usage: {end_mem / 1024 / 1024:.2f} MB")
print(f"Peak GPU memory usage: {end_gpu_mem / 1024 / 1024:.2f} MB")
