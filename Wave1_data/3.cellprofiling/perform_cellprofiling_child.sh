#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --partition=amilan
#SBATCH --qos=normal
#SBATCH --account=amc-general
#SBATCH --time=20:00
#SBATCH --output=cp_child-%j.out

# 2 cores at 3.75 GB of ram per core

# activate cellprofiler environment
module load anaconda
conda init bash
conda activate cellprofiler_timelapse_env

dir=$1

jupyter nbconvert --to=script --FilesWriter.build_directory=scripts/ notebooks/*.ipynb

cd scripts/ || exit

python run_cellprofiler_analysis.py --input_dir "$dir"

cd .. || exit

# deactivate cellprofiler environment
conda deactivate

echo "Cellprofiler analysis done"
