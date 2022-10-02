#!/bin/bash

#SBATCH -J MI                # Job name
#SBATCH -o .out/mi_%a.out   # Name of stdout output file (%j expands to %jobID)
#SBATCH --ntasks=60
#SBATCH --mem-per-cpu=2000
#SBATCH --cpus-per-task=1

echo "Monkey: $1";

# TT, BR, ALIGN, AVG, MONKEY, DELAY
python -O mi_power_analysis.py 1 1 "cue" 1 "lucy" 0
python -O mi_power_analysis.py 1 1 "cue" 1 "lucy" 1
python -O mi_power_analysis.py 1 1 "cue" 0 "$1" 0
