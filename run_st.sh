#!/bin/bash

#SBATCH -J COH                # Job name
#SBATCH -o coh.out            # Name of stdout output file (%j expands to %jobID)
#SBATCH -t 700:00:00          # Run time (hh:mm:ss) - 1.5 hours
#SBATCH -n 40
#SBATCH -N 1
##SBATCH --array=1:5

python3 save_super_tensors.py