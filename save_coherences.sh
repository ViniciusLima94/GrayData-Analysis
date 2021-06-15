#!/bin/bash

#SBATCH -J COH                # Job name
#SBATCH -o coh.out            # Name of stdout output file (%j expands to %jobID)
#SBATCH --ntasks=40
#SBATCH --mem-per-cpu=2000
#SBATCH --cpus-per-task=1
#SBATCH --array=0-64

#module load python/3.6.8
#python3 -O save_coherences.py $SLURM_ARRAY_TASK_ID
python -O save_coherences.py $SLURM_ARRAY_TASK_ID
#python3 -O save_coherences.py
