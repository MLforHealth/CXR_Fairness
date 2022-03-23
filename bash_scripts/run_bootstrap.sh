#!/bin/bash

set -e
cd ../

slurm_pre="--partition cpu --mem 100gb -c 15 --qos nopreemption --job-name bootstrap --output /scratch/ssd001/home/haoran/projects/CXR_Fairness/logs/bootstrap_%A.log"

python -m cxr_fairness.sweep launch \
    --experiment Bootstrap \
    --slurm_pre "${slurm_pre}" \
    --command_launcher "slurm" \
    --no_output_dir 
