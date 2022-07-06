#!/bin/bash

#SBATCH -J GC               # Job name
#SBATCH -o .out/GC_%a.out   # Name of stdout output file (%j expands to %jobID)
#SBATCH --ntasks=40
#SBATCH --mem-per-cpu=2000
#SBATCH --cpus-per-task=1
#SBATCH --array=0-61

python -O ghost_coherence.py $SLURM_ARRAY_TASK_ID "coh"
