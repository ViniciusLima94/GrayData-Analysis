#!/bin/bash

#SBATCH -J coh               # Job name
#SBATCH -o .out/coh_%a.out   # Name of stdout output file (%j expands to %jobID)
#SBATCH --ntasks=40
#SBATCH --mem-per-cpu=2000
#SBATCH --cpus-per-task=1
#SBATCH --array=0-61

python -O spectral_content.py $SLURM_ARRAY_TASK_ID
#python -O find_peaks.py $SLURM_ARRAY_TASK_ID

