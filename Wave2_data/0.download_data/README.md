# Download data
In the case that this data becomes publically available, this section will be updated with instructions on how to download the data.

A bit about the information though, in case it could be of use to others.

The raw data are:
- 1 - 384-well plate
    - 56 wells were used
- 4 - FOVs per well
- 5 - channels per FOV
- 1 - z-plane per FOV
- Acquired every 15 minutes for 20 hours (81 time points)

For a total of 4 * 5 * 56 * 81 =  images. Each image is a 16-bit TIFF file, and the total size of the raw data is approximately 853 GB.

Transferring from a google cloud bucket took roughly 12 hours, and the data was transferred to a local hard drive for processing.
The data was then processed on a local workstation with 128 GB of RAM and an NVIDIA RTX 3090 GPU.
