#!/bin/bash
#SBATCH --nodes=1
#SBATCH --partition=amilan
#SBATCH --qos=normal
#SBATCH --account=amc-general
#SBATCH --time=10:00
#SBATCH --output=ic_parent-%j.out

# This script runs Illumination Correction on the raw image data.
module load anaconda
conda init bash
conda activate timelapse_basicpy_env

loadfile_path="./loadfiles/well_fovs_to_run.tsv"

readarray -t well_fovs_to_run < <(tail -n +2 "$loadfile_path" | cut -f1)

jupyter nbconvert --to=script --FilesWriter.build_directory=scripts/ notebooks/*.ipynb


for well_fov in "${well_fovs_to_run[@]}"; do
	# get the number of jobs for the user
    number_of_jobs=$(squeue -u $USER | wc -l)
    while [ $number_of_jobs -gt 990 ]; do
        sleep 1s
        number_of_jobs=$(squeue -u $USER | wc -l)
    done
    sbatch run_ic_child_HPC.sh "$well_fov"
done

conda deactivate

echo "All illumination correction jobs submitted!"
