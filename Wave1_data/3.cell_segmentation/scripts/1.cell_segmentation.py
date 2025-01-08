#!/usr/bin/env python
# coding: utf-8

# This notebook focuses on trying to find a way to segment cells within organoids properly.
# The end goals is to segment cell and extract morphology features from cellprofiler.
# These masks must be imported into cellprofiler to extract features.

# In[1]:


import argparse
import pathlib

import matplotlib.pyplot as plt

# Import dependencies
import numpy as np
import skimage
import tifffile
from cellpose import models
from PIL import Image
from stardist.plot import render_label

# check if in a jupyter notebook
try:
    cfg = get_ipython().config
    in_notebook = True
except NameError:
    in_notebook = False

print(in_notebook)


# In[2]:


if not in_notebook:
    # set up arg parser
    parser = argparse.ArgumentParser(description="Segment the nuclei of a tiff image")

    parser.add_argument(
        "--input_dir",
        type=str,
        help="Path to the input directory containing the tiff images",
    )

    parser.add_argument(
        "--clip_limit",
        type=float,
        help="Clip limit for the adaptive histogram equalization",
    )
    parser.add_argument(
        "--optimize_segmentation",
        action="store_true",
        help="Optimize the segmentation parameters",
    )

    args = parser.parse_args()
    clip_limit = args.clip_limit
    input_dir = pathlib.Path(args.input_dir).resolve(strict=True)
    optimize_segmentation = args.optimize_segmentation

else:
    input_dir = pathlib.Path(
        "../../2.illumination_correction/illum_directory/W0078_F0001"
    ).resolve(strict=True)
    clip_limit = 0.4
    optimize_segmentation = True


figures_dir = pathlib.Path("../figures").resolve()
figures_dir.mkdir(exist_ok=True, parents=True)


# ## Set up images, paths and functions

# In[3]:


image_extensions = {".tif", ".tiff"}
files = sorted(input_dir.glob("*"))
files = [str(x) for x in files if x.suffix in image_extensions]


# In[4]:


image_dict = {
    "nuclei_file_paths": [],
    "nuclei": [],
    "cytoplasm1": [],
    "cytoplasm2": [],
    "cytoplasm3": [],
}


# In[5]:


# split files by channel
for file in files:
    if "C4" in file.split("/")[-1]:
        image_dict["nuclei_file_paths"].append(file)
        image_dict["nuclei"].append(tifffile.imread(file).astype(np.float32))
    elif "C2" in file.split("/")[-1]:
        image_dict["cytoplasm1"].append(tifffile.imread(file).astype(np.float32))
    elif "C3" in file.split("/")[-1]:
        image_dict["cytoplasm2"].append(tifffile.imread(file).astype(np.float32))

cytoplasm_image_list = [
    np.max(
        np.array(
            [
                cytoplasm1,
                cytoplasm2,
            ]
        ),
        axis=0,
    )
    for cytoplasm1, cytoplasm2 in zip(
        image_dict["cytoplasm1"],
        image_dict["cytoplasm2"],
    )
]
nuclei_image_list = [np.array(nuclei) for nuclei in image_dict["nuclei"]]

cyto = np.array(cytoplasm_image_list).astype(np.int16)
nuclei = np.array(nuclei_image_list).astype(np.int16)

cyto = skimage.exposure.equalize_adapthist(cyto, clip_limit=clip_limit + 0.3)
nuclei = skimage.exposure.equalize_adapthist(nuclei, clip_limit=clip_limit)


print(cyto.shape, nuclei.shape)


# In[6]:


original_nuclei_image = nuclei.copy()
original_cyto_image = cyto.copy()


# In[7]:


imgs = []
# save each z-slice as an RGB png
for z in range(nuclei.shape[0]):

    nuclei_tmp = nuclei[z, :, :]
    cyto_tmp = cyto[z, :, :]
    nuclei_tmp = (nuclei_tmp / nuclei_tmp.max() * 255).astype(np.uint8)
    cyto_tmp = (cyto_tmp / cyto_tmp.max() * 255).astype(np.uint8)
    # save the image as an RGB png with nuclei in blue and cytoplasm in red
    RGB = np.stack([cyto_tmp, np.zeros_like(cyto_tmp), nuclei_tmp], axis=-1)

    # change to 8-bit
    RGB = (RGB / RGB.max() * 255).astype(np.uint8)

    rgb_image_pil = Image.fromarray(RGB)

    imgs.append(rgb_image_pil)


# ## Cellpose

# In[ ]:


if optimize_segmentation:
    # model_type='cyto' or 'nuclei' or 'cyto2' or 'cyto3'
    model_name = "cyto3"
    model = models.Cellpose(model_type=model_name, gpu=True)

    channels = [[1, 3]]  # channels=[red cells, blue nuclei]

    masks_all_dict = {"masks": [], "imgs": [], "diameter": []}
    # get masks for all the images
    # save to a dict for later use
    frame = 1
    imgs = np.array(imgs)
    img = imgs[frame, :, :]
    # img = normalize(img)
    for diameter in range(10, 200, 5):
        masks, flows, styles, diams = model.eval(
            img, channels=channels, diameter=diameter
        )
        masks_all_dict["masks"].append(masks)
        masks_all_dict["imgs"].append(img)
        masks_all_dict["diameter"].append(diameter)
    print(len(masks_all_dict))
    masks_all = masks_all_dict["masks"]
    imgs = masks_all_dict["imgs"]
    diameters = masks_all_dict["diameter"]

    masks_all = np.array(masks_all)
    imgs = np.array(imgs)

    if in_notebook:
        for diameter in range(len(diameters)):
            plt.figure(figsize=(20, 6))
            plt.title(f"Diameter: {diameters[diameter]}")
            plt.axis("off")
            plt.subplot(1, 4, 1)
            plt.imshow(nuclei[frame], cmap="gray")
            plt.title("Nuclei")
            plt.axis("off")

            plt.subplot(1, 4, 2)
            plt.imshow(imgs[diameter])
            plt.title("Red: Cell, Blue: Nuclei")
            plt.axis("off")

            plt.subplot(1, 4, 3)
            plt.imshow(render_label(masks_all[diameter]))
            plt.title("Cell masks")
            plt.axis("off")

            plt.subplot(1, 4, 4)
            plt.imshow(cyto[frame], cmap="gray")
            plt.title("Cytoplasm")
            plt.axis("off")
        plt.show()

    # get the number of unique masks for each diameter
    unique_masks_dict = {
        "diameter": [],
        "unique_masks": [],
    }
    for diameter in range(len(diameters)):
        unique_masks = np.unique(masks_all[diameter])
        unique_masks_dict["diameter"].append(diameters[diameter])
        unique_masks_dict["unique_masks"].append(len(unique_masks))

    # get the idex of the largest number
    np.max(unique_masks_dict["unique_masks"])
    # get the diameter that is for the max
    best_diameter = unique_masks_dict["diameter"][
        np.argmax(unique_masks_dict["unique_masks"])
    ]

    plt.plot(unique_masks_dict["diameter"], unique_masks_dict["unique_masks"])
    # plot a vertical line
    plt.axvline(x=best_diameter, color="red", linestyle="--")
    plt.xlabel("Diameter")
    plt.ylabel("Number of unique masks")
    plt.title("Number of unique masks vs Diameter")
    if in_notebook:
        plt.show()
    plt.savefig("../figures/unique_cell_masks_vs_diameter.png")


# In[ ]:


if not optimize_segmentation:
    # model_type='cyto' or 'nuclei' or 'cyto2' or 'cyto3'
    model_name = "cyto3"
    model = models.Cellpose(model_type=model_name, gpu=True)

    channels = [[1, 3]]  # channels=[red cells, blue nuclei]

    masks_all_dict = {"masks": [], "imgs": []}
    imgs = np.array(imgs)

    # get masks for all the images
    # save to a dict for later use
    for img in imgs:
        # masks, flows, styles, diams = model.eval(img, diameter=diameter, channels=channels)
        masks, flows, styles, diams = model.eval(img, channels=channels, diameter=100)

        masks_all_dict["masks"].append(masks)
        masks_all_dict["imgs"].append(img)
    print(len(masks_all_dict))
    masks_all = masks_all_dict["masks"]
    imgs = masks_all_dict["imgs"]

    # save the generated masks
    for frame_index, frame in enumerate(image_dict["nuclei_file_paths"]):
        tifffile.imwrite(
            f"{input_dir}/{str(frame).split('/')[-1].split('_C4')[0]}_cell_mask.tiff",
            masks_all[frame_index],
        )

    if in_notebook:
        for z in range(len(masks_all)):
            plt.figure(figsize=(30, 10))
            plt.title(f"z: {z}")
            plt.axis("off")
            plt.subplot(1, 4, 1)
            plt.imshow(nuclei[z], cmap="gray")
            plt.title("Nuclei")
            plt.axis("off")

            plt.subplot(142)
            plt.imshow(cyto[z], cmap="gray")
            plt.title("Cytoplasm")
            plt.axis("off")

            plt.subplot(143)
            plt.imshow(imgs[z], cmap="gray")
            plt.title("Red: Cytoplasm, Blue: Nuclei")
            plt.axis("off")

            plt.subplot(144)
            plt.imshow(render_label(masks_all[z]))
            plt.title("Cell masks")
            plt.axis("off")
            plt.show()
