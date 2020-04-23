#!/bin/bash

#SBATCH -J COH                # Job name
#SBATCH -o coh.out            # Name of stdout output file (%j expands to %jobID)
#SBATCH -t 700:00:00          # Run time (hh:mm:ss) - 1.5 hours
#SBATCH -N 1 
##SBATCH --exclude=c[01-04],clusterneuromat
#SBATCH --array=1,2,3,4,5


python3 save_coherences.py $SLURM_ARRAY_TASK_ID
