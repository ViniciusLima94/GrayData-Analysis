#!/bin/bash

#SBATCH -J BST                # Job name
#SBATCH -o bst.out            # Name of stdout output file (%j expands to %jobID)
#SBATCH -t 700:00:00          # Run time (hh:mm:ss) - 1.5 hours
#SBATCH -n 40
#SBATCH -N 1
#SBATCH --array=0-25

python3.6 -O burstness_analysis.py $SLURM_ARRAY_TASK_ID
