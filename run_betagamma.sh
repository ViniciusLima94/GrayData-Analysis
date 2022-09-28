#!/bin/bash

#SBATCH -J BG               # Job name
#SBATCH -o .out/BG_%a.out   # Name of stdout output file (%j expands to %jobID)
#SBATCH --ntasks=5
#SBATCH --mem-per-cpu=2000
#SBATCH --cpus-per-task=1
#SBATCH --array=0-61

python -O beta_gamma_corr.py $SLURM_ARRAY_TASK_ID
