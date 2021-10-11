#!/bin/bash

#SBATCH -J BST                # Job name
#SBATCH -o bst.out            # Name of stdout output file (%j expands to %jobID)
#SBATCH --ntasks=40
#SBATCH --mem-per-cpu=2000
#SBATCH --cpus-per-task=1
#SBATCH --array=0-3

python -O save_burst_stats.py "multitaper" $SLURM_ARRAY_TASK_ID
