#!/bin/bash

#SBATCH -J POW                # Job name
#SBATCH -o .out/pow_%a.out   # Name of stdout output file (%j expands to %jobID)
#SBATCH --ntasks=40
#SBATCH --mem-per-cpu=2000
#SBATCH --cpus-per-task=1
#SBATCH --array=33-64

python -O save_power.py $SLURM_ARRAY_TASK_ID