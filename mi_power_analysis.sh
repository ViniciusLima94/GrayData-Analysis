#!/bin/bash

#SBATCH -J MI                # Job name
#SBATCH -o .out/mi_%a.out   # Name of stdout output file (%j expands to %jobID)
#SBATCH --ntasks=40
#SBATCH --mem-per-cpu=2000
#SBATCH --cpus-per-task=1

python -O mi_power_analysis.py 1 1 "cue" 1 
