#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=amilan
#SBATCH --qos=normal
#SBATCH --account=amc-general
#SBATCH --time=60:00
#SBATCH --output=cp_parent-%j.out

# 2 cores at 3.75 GB of ram per core

# activate cellprofiler environment
module load anaconda
conda init bash
conda activate cellprofiler_timelapse_env

jupyter nbconvert --to=script --FilesWriter.build_directory=scripts/ notebooks/*.ipynb

cd scripts/ || exit

# get a list of all dirs in the raw data folder
data_dir="../../2.illumination_correction/illum_directory"
# data_dir="../../../data/test_dir"
mapfile -t FOV_dirs < <(ls -d $data_dir/*)
echo length of plate_dirs: ${#FOV_dirs[@]}

cd ../ || exit


for FOV_dir in "${FOV_dirs[@]}"; do
	# get the number of jobs for the user
    number_of_jobs=$(squeue -u $USER | wc -l)
    while [ $number_of_jobs -gt 990 ]; do
        sleep 1s
        number_of_jobs=$(squeue -u $USER | wc -l)
    done
    sbatch perform_cellprofiling_child.sh "$FOV_dir"
done

conda deactivate

echo "All cellprofiling jobs submitted!"


