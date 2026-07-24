#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=1:00:00
#SBATCH --partition=amilan
#SBATCH --output=cell_tracking-%j.out


jupyter nbconvert --to=script --FilesWriter.build_directory=scripts/ notebooks/*.ipynb


# check if the plate_name argument is provided
if [ -z "$plate_name" ]; then
    echo "Error: No plate name provided."
    echo "using default plate name: 'plate_2'"
    plate_name="plate_2"
fi

cd scripts/ || exit

python 0.generate_load_file.py --plate_name "$plate_name"
load_file_path="../loadfiles/loadfile.txt"

while IFS= read -r well_fov; do
    echo "Well FOV: $well_fov submitting to HPC_child_cell_tracking_script.sh"
    sbatch HPC_child_cell_tracking_script.sh "$plate_name" "$well_fov"
done < "$load_file_path"


cd ../ || exit

conda deactivate

echo "Cell tracking script completed"
