#!/bin/bash

#SBATCH -J COH                # Job name
#SBATCH -o coh.out            # Name of stdout output file (%j expands to %jobID)
#SBATCH --ntasks=40
#SBATCH --mem-per-cpu=2000
#SBATCH --cpus-per-task=1
#SBATCH --array=0-47

python -O plot_average_coherence.py $SLURM_ARRAY_TASK_ID
