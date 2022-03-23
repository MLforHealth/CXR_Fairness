#!/bin/bash

set -e
cd ../

slurm_pre="--partition t4v2,t4v1,rtx6000 --gres gpu:1 --mem 25gb -c 4 --job-name ${1} --output /scratch/ssd001/home/haoran/projects/CXR_Fairness/logs/${1}_%A.log"

output_root="/scratch/ssd001/home/haoran/cxr_debias"

python -m cxr_fairness.sweep launch \
    --experiment ${1} \
    --output_root "${output_root}" \
    --slurm_pre "${slurm_pre}" \
    --command_launcher "slurm" \
    --max_slurm_jobs 200
