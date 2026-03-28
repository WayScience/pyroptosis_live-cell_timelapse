#!/bin/bash

# activate the preprocessing environment
conda activate cellprofiler_timelapse_env

jupyter nbconvert --to script --output-dir=scripts/ notebooks/*.ipynb


cd scripts/ || exit

python generate_load_data.py
python run_cellprofiler_analysis.py --max_workers 10

conda deactivate

echo "Cell segmentation preprocessing completed successfully."
