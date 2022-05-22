#!/bin/bash

#SBATCH -J MI                # Job name
#SBATCH -o .out/net_%a.out   # Name of stdout output file (%j expands to %jobID)
#SBATCH --ntasks=60
#SBATCH --mem-per-cpu=2000
#SBATCH --cpus-per-task=1
#SBATCH --array=29-61

python -O save_network_analysis.py "coh" $SLURM_ARRAY_TASK_ID "cue"
