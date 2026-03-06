#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=amilan
#SBATCH --qos=normal
#SBATCH --account=amc-general
#SBATCH --time=1:00:00
#SBATCH --output=segmentation_parent-%j.out

module load anaconda
conda activate timelapse_segmentation_env

jupyter nbconvert --to=script --FilesWriter.build_directory=scripts/ notebooks/*.ipynb

cd scripts || exit

loadfile_path="../loadfiles/loadfile.txt"

while IFS= read -r well_fov; do

    number_of_jobs=$(squeue -u $USER | wc -l)
        while [ $number_of_jobs -gt 990 ]; do
            sleep 1s
            number_of_jobs=$(squeue -u $USER | wc -l)
        done
        sbatch HPC_run_segmentation_child.sh "$well_fov"

done < "$loadfile_path"

cd .. || exit

conda deactivate

echo "Segmentation submitted to HPC"
