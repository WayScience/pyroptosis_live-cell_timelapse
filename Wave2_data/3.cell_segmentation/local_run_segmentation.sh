#!/bin/bash

conda activate timelapse_segmentation_env

jupyter nbconvert --to=script --FilesWriter.build_directory=scripts/ notebooks/*.ipynb

cd scripts || exit

loadfile_path="../loadfiles/loadfile.txt"

while IFS= read -r well_fov; do
    echo "Processing $well_fov"
    python 1.nuclei_segmentation.py --well_fov "$well_fov" --clip_limit 0.6
    python 2.cell_segmentation.py --well_fov "$well_fov" --clip_limit 0.3
done < "$loadfile_path"

cd .. || exit

conda deactivate

echo "Segmentation done"
