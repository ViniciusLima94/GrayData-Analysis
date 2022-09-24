#!/bin/bash

#SBATCH -J MC                # Job name
#SBATCH -o .out/MC_%a.out    # Name of stdout output file (%j expands to %jobID)
#SBATCH --ntasks=40
#SBATCH --mem-per-cpu=2000
#SBATCH --cpus-per-task=1
#SBATCH --array=0-61

# Original
                       # SIDX           METRIC SURR THR MONKEY
python -O meta_conn.py $SLURM_ARRAY_TASK_ID "coh" 0 0 "lucy"
# Surrogate
python -O meta_conn.py $SLURM_ARRAY_TASK_ID "coh" 1 0 "lucy"
# Surrogate strong
python -O meta_conn.py $SLURM_ARRAY_TASK_ID "coh" 2 0 "lucy"

