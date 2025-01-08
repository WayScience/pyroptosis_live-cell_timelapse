#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=amilan
#SBATCH --qos=long
#SBATCH --account=amc-general
#SBATCH --time=7-00:00:00
#SBATCH --output=segmentation_parent-%j.out

module load anaconda

conda activate timelapse_segmentation_env

jupyter nbconvert --to=script --FilesWriter.build_directory=scripts/ notebooks/*.ipynb

cd scripts || exit

illumination_correction_dir="../../2.illumination_correction/illum_directory"

mapfile -t dirs < <(ls -d "$illumination_correction_dir"/*)

cd .. || exit

touch job_ids.out
jobs_submitted_counter=0
for dir in "${dirs[@]}"; do
    echo "$dir"
    number_of_jobs=$(squeue -u $USER | wc -l)
        while [ $number_of_jobs -gt 990 ]; do
            sleep 1s
            number_of_jobs=$(squeue -u $USER | wc -l)
        done

        job_id=$(sbatch HPC_run_segmentation_child.sh "$dir")
        # append the job id to the file
        job_id=$(echo $job_id | awk '{print $4}')
        echo " '$job_id' '$dir' "
        echo " '$job_id' '$dir' " >> job_ids.out
        let jobs_submitted_counter++
done

cd .. || exit

conda deactivate

echo "Segmentation done"
