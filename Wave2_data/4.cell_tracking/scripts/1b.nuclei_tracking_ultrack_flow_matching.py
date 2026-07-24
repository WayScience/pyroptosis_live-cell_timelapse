#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import pathlib

import dask.array as da
import napari
import natsort
import numpy as np
import pandas as pd
import scipy.ndimage as ndi
import tifffile
from napari.utils.notebook_display import nbscreenshot
from rich.pretty import pprint
from tifffile import imread
from timelapse_utils.file_utils.notebook_init_utils import (
    bandicoot_check,
    init_notebook,
)
from ultrack import (
    MainConfig,
    add_flow,
    link,
    segment,
    solve,
    to_ctc,
    to_tracks_layer,
    tracks_to_zarr,
)
from ultrack.imgproc import detect_foreground, robust_invert
from ultrack.imgproc.flow import (
    advenct_from_quasi_random,
    timelapse_flow,
    trajectories_to_tracks,
)
from ultrack.tracks.stats import tracks_df_movement
from ultrack.utils.array import array_apply, create_zarr
from ultrack.utils.cuda import on_gpu

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
# with open(config_output_path, "r") as f:
#     config_dict = json.load(f)
# config = MainConfig.parse_obj(config_dict)

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
for tiff_file in mask_files[:50]:
    img = tifffile.imread(tiff_file)
    masks.append(img)

masks = np.array(masks)

nuclei = []
for tiff_file in nuclei_files[:50]:
    img = tifffile.imread(tiff_file)
    nuclei.append(img)
nuclei = np.array(nuclei)


# In[6]:


image_dims = nuclei[0].shape


# In[7]:


image = masks

viewer = napari.Viewer()

im_layer = viewer.add_image(nuclei)
image = viewer.layers[0].data


# In[8]:


# detection = create_zarr(image.shape, bool, "detection.zarr", overwrite=True)
# array_apply(
#     image,
#     out_array=detection,
#     func=on_gpu(detect_foreground),
# )

# viewer.add_labels(detection, visible=True).contour = 2
detection = masks
viewer.add_labels(detection, visible=True)


# In[9]:


boundaries = create_zarr(image.shape, np.float16, "boundaries.zarr", overwrite=True)
array_apply(
    image,
    out_array=boundaries,
    func=on_gpu(robust_invert),
    sigma=3.0,
)

viewer.add_image(boundaries, visible=False)


# In[10]:


if pathlib.Path("flow.zarr").exists():
    get_ipython().system("rm -r flow.zarr # removing previous flow")
flow = timelapse_flow(
    image, store_or_path="flow.zarr", n_scales=2, lr=1e-2, num_iterations=2_000
)
viewer.add_image(
    flow,
    contrast_limits=(-0.001, 0.001),
    colormap="turbo",
    visible=False,
    scale=(4,) * 3,
    channel_axis=1,
    name="flow field",
)


# In[11]:


trajectory = advenct_from_quasi_random(flow, detection.shape[-2:], n_samples=1000)
flow_tracklets = pd.DataFrame(
    trajectories_to_tracks(trajectory),
    columns=["track_id", "t", "y", "x"],
)
flow_tracklets[
    ["y", "x"]
] += 0.5  # napari was crashing otherwise, might be an openGL issue
flow_tracklets[["dy", "dx"]] = tracks_df_movement(flow_tracklets)
flow_tracklets["angles"] = np.arctan2(flow_tracklets["dy"], flow_tracklets["dx"])

flow_tracklets.to_csv("flow_tracklets.csv", index=False)

viewer.add_tracks(
    flow_tracklets[["track_id", "t", "y", "x"]],
    name="flow vectors",
    visible=True,
    tail_length=25,
    features=flow_tracklets[["angles", "dy", "dx"]],
    colormap="hsv",
).color_by = "angles"

# nbscreenshot(viewer)


# In[12]:


cfg = MainConfig()

cfg.data_config.n_workers = 1

cfg.segmentation_config.n_workers = 1
cfg.segmentation_config.min_area = 250
cfg.segmentation_config.max_area = 15_000

cfg.linking_config.n_workers = 1
cfg.linking_config.max_neighbors = 5
cfg.linking_config.max_distance = 20.0

cfg.tracking_config.window_size = 70
cfg.tracking_config.overlap_size = 5
cfg.tracking_config.appear_weight = -0.01
cfg.tracking_config.disappear_weight = -0.001
cfg.tracking_config.division_weight = 0

pprint(cfg)


# In[13]:


segment(detection, boundaries, cfg, overwrite=True)
add_flow(cfg, flow)


# In[14]:


import pandas as pd

pd.options.mode.copy_on_write = False
import numpy as np
import skimage.util._map_array as map_array_mod

_original_map_array = map_array_mod.map_array


def _patched_map_array(input_arr, input_vals, output_vals, out=None):
    input_arr = np.array(input_arr, copy=True)
    input_vals = np.array(input_vals, copy=True)
    output_vals = np.array(output_vals, copy=True)
    return _original_map_array(input_arr, input_vals, output_vals, out=out)


map_array_mod.map_array = _patched_map_array


# In[15]:


link(cfg, overwrite=True)
solve(cfg)


# In[16]:


tracks_df, graph = to_tracks_layer(cfg)
tracks_df.to_csv("tracks.csv", index=False)

segments = tracks_to_zarr(
    cfg,
    tracks_df,
    store_or_path="segments.zarr",
    overwrite=True,
)

viewer.layers["flow vectors"].visible = False
# viewer.layers["detection"].visible = False
viewer.add_tracks(
    tracks_df[["track_id", "t", "y", "x"]],
    name="tracks",
    graph=graph,
    visible=True,
)

viewer.add_labels(da.from_zarr(segments), name="segments").contour = 2

# nbscreenshot(viewer)


# In[17]:


tracks_df["track_id"].unique()


# In[18]:


np.unique(masks)
