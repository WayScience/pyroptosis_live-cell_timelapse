#!/bin/bash

conda activate cellprofiler_timelapse_env
jupyter nbconvert --to=script --FilesWriter.build_directory=scripts/ notebooks/*.ipynb

cd scripts || exit

loadfile_path="../loadfiles/well_fovs_to_run.tsv"
counter=0
total_lines=$(wc -l < "${loadfile_path}")
echo "Total images to process: ${total_lines}"
while IFS=$'\t' read -r well_fov; do
    # run the Python script with the current well_fov as an argument
    # takes about 15 min per 102 timepoints for 5 channels
    # which this script runs
    python 1.perform_ic_cp.py --well_fov "${well_fov}"
    # update the counter and print progress
    counter=$((counter + 1))
    echo "Processed ${counter} of ${total_lines} images"

done < "${loadfile_path}"

echo "Finished running IC for all images."
