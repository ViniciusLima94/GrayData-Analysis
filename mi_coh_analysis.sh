#!/bin/bash

#SBATCH -J MI                # Job name
#SBATCH -o .out/mi_coh_%a.out   # Name of stdout output file (%j expands to %jobID)
#SBATCH --ntasks=40
#SBATCH --mem-per-cpu=2000
#SBATCH --cpus-per-task=1
#SBATCH --array=0-64

python -O mi_coh_analysis.py $SLURM_ARRAY_TASK_ID