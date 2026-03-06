#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --partition=aa100
#SBATCH --gres=gpu:1
#SBATCH --qos=normal
#SBATCH --account=amc-general
#SBATCH --time=15:00
#SBATCH --output=segmentation_child-%j.out

module load anaconda

conda activate timelapse_segmentation_env

cd scripts || exit

well_fov=$1

echo "$well_fov"

python 1.nuclei_segmentation.py --well_fov "$well_fov" --clip_limit 0.6
python 2.cell_segmentation.py --well_fov "$well_fov" --clip_limit 0.3

cd .. || exit

conda deactivate

echo "Segmentation done"
