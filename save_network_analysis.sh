#!/bin/bash

#SBATCH -J MI                # Job name
#SBATCH -o .out/net_%a.out   # Name of stdout output file (%j expands to %jobID)
#SBATCH --ntasks=40
#SBATCH --mem-per-cpu=2000
#SBATCH --cpus-per-task=1
#SBATCH --array=0-61

python -O save_network_analysis.py "plv" $SLURM_ARRAY_TASK_ID 1 "cue"
