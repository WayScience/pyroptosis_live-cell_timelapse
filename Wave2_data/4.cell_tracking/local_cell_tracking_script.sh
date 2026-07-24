#!/bin/bash

conda activate timelapse_tracking_env


jupyter nbconvert --to=script --FilesWriter.build_directory=scripts/ notebooks/*.ipynb
plate_name=$1

cd scripts/ || exit
python 0.generate_load_file.py --plate_name "$plate_name"
load_file_path="../loadfiles/loadfile.txt"

while IFS= read -r well_fov; do
    echo "Processing $well_fov"
    python 1c.nuclei_tracking_HOCT.py --well_fov "$well_fov" --plate_name "$plate_name"
# done < "$load_file_path"
done < <(tac "$load_file_path")  # Reverse the order of lines in loadfile.txt


cd ../ || exit

conda deactivate

echo "Cell tracking script completed"
