#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=amilan
#SBATCH --qos=normal
#SBATCH --account=amc-general
#SBATCH --time=0:10:00
#SBATCH --output=ic_child-%j.out

# This script runs Illumination Correction on the raw image data.
module load anaconda
conda init bash
conda activate timelapse_basicpy_env

well_fov="$1"
echo "Processing $well_fov"
cd scripts/ || exit

python 1.perform_ic.py --well_fov "$well_fov"

cd ../ || exit

conda deactivate

echo "Illumination Correction complete!"

