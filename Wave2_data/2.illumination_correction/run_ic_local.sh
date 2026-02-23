#!/bin/bash
# This script runs Illumination Correction on the raw image data.
conda activate timelapse_basicpy_env

jupyter nbconvert --to python --output-dir=scripts/ notebooks/*.ipynb
cd scripts/ || exit
loadfile_path="../loadfiles/well_fovs_to_run.tsv"

readarray -t well_fovs_to_run < <(tail -n +2 "$loadfile_path" | cut -f1)

for well_fov in "${well_fovs_to_run[@]}"; do
    echo "Processing $well_fov"
    python 1.perform_ic.py --well_fov "$well_fov"
done

cd ../ || exit

conda deactivate

echo "Illumination Correction complete!"
