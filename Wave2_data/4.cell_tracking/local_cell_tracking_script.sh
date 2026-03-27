#!/bin/bash

conda activate timelapse_segmentation_env


jupyter nbconvert --to=script --FilesWriter.build_directory=scripts/ notebooks/*.ipynb

cd scripts/ || exit

load_file_path="../loadfiles/loadfile.txt"

while IFS= read -r well_fov; do
    echo "Processing $well_fov"
    python 0.nuclei_tracking.py --well_fov "$well_fov"
done < "$load_file_path"


cd ../ || exit

conda deactivate

echo "Cell tracking script completed"
