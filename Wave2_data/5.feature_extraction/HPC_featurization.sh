#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=128
#SBATCH --partition=amilan
#SBATCH --constraint=128c # specify the constraint for 128 cores
#SBATCH --qos=long
#SBATCH --account=amc-general
#SBATCH --time=5:00:00
#SBATCH --output=timelapse_cellprofiling-%j.out

# 2 cores at 3.75 GB of ram per core

# activate cellprofiler environment
module load anaconda
conda init bash
# activate the preprocessing environment
conda activate cellprofiler_timelapse_env

jupyter nbconvert --to script --output-dir=scripts/ notebooks/*.ipynb


cd scripts/ || exit

python generate_load_data.py
# @120 workers this should run each well_fov in parallel (102 timepoints)
# we have 224 well fovs (224 x 102) = 22848 total timepoints to process,
# 22848@ ~1.5 minutes per timepoint is 34,272 minutes or 571.2 hours
# 34727 minutes/126 workers = 275.61 minutes or 4.59 hours
# 126 to buffer ram usage,
# 128 cores available but we want to leave some overhead for the system

python run_cellprofiler_analysis.py --max_workers 126

conda deactivate

echo "Cell segmentation preprocessing completed successfully."

