#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=5
#SBATCH --time=2:00:00
#SBATCH --partition=aa100
#SBATCH --qos=gpu-normal
#SBATCH --gres=gpu:a100-40gb:1
#SBATCH --output=cell_tracking-%j.out

module load cuda/11.8
module load uv
git_root=$(git rev-parse --show-toplevel)

if [ -d "/scratch/alpine" ]; then
    echo "Using Alpine environment"
    ENV_PATH="/projects/mlippincott@xsede.org/software/uv/envs/timelapse_live_cell_pyroptosis_uv_env/.venv/bin/activate"
elif [ -d "/anvil" ]; then
    ENV_PATH="/anvil/projects/x-bio260064/software/uv/envs/timelapse_live_cell_pyroptosis_uv_env/.venv/bin/activate"
else
    ENV_PATH="$git_root/.venv/bin/activate"
fi

PYTHON_BIN="$ENV_PATH/bin/python3"

source "$ENV_PATH"
cd scripts/ || exit

plate_name=$1
well_fov=$2

pyhthon 1c.nuclei_tracking_HOCT.py --well_fov "$well_fov" --plate_name "$plate_name"

cd ../ || exit

conda deactivate

echo "Cell tracking script completed"
