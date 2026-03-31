#!/bin/bash
# establish the git root and load the list of well_fov_times to process
git_root=$(git rev-parse --show-toplevel)
# establish the load data and load it into an array
load_data_file_path="${git_root}/Wave2_data/7.image_based_profiling/load_data/load_file.txt"

readarray -t well_fov_times < "$load_data_file_path"


conda activate timelapse_ibp_env

# Optional: regenerate scripts from notebooks
jupyter nbconvert --to=script --FilesWriter.build_directory=scripts/ notebooks/*.ipynb

cd scripts || exit

python 00.generate_load_list.py
python 0.merge_sc.py
python 1.combine_sc.py
python 2.qc.py
python 3.annotate_sc.py
# python 5.single_cell_track_merging_placeholder.py
python 6.normalize_sc.py
python 7.feature_select_sc.py
python 8.aggregate_profiles.py

conda deactivate ; conda activate timelapse_deeplearning_env
for well_fov_time in "${well_fov_times[@]}"; do
    echo "Featurizing for well_fov_time: $well_fov_time"
    python 4a.chammi75_featurization.py --well_fov_time "$well_fov_time"
done
python 4b.chammi75_combine_sc.py

conda deactivate ; conda activate timelapse_ibp_env

python 9.harmonize_profiles.py

conda deactivate
cd ../  || exit
