# Feature Extraction

## Overview
This directory contains scripts to run cellprofiler for each well_fov_timepoint and extract features from the timelapse data.

### Deployment strategy
Run each well_fov_timepoint in parallel.
We have 224 well fovs and 102 timepoints per well fov, so we have 22848 total timepoints to process.
Each timepoint takes ~1.5 minutes to process, so the total time to process all timepoints is 34,272 minutes or 571.2 hours.
With 126 on HPC workers, we can process all timepoints in 275.61 minutes or 4.59 hours.
With ~10-24 workers on a local machine, we can process all timepoints in 3,427-1,428 minutes or 57.1-23.8 hours, respectively.

## Running the analysis
```
source run_local_featurization.sh
```
or
```
sbatch HPC_featurization.sh
```
