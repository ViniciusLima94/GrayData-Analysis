#!/bin/bash

#SBATCH -J MI                # Job name
#SBATCH -o .out/mi_coh_%a.out   # Name of stdout output file
#SBATCH --ntasks=40
#SBATCH --mem-per-cpu=2000
#SBATCH --cpus-per-task=1

echo "Monkey: $1";

# METRIC AVERAGED SURR MONKEY ALIGNED DELAY
python -O mi_coh_analysis.py "coh" 1 0 "$1" "cue" 0
python -O mi_coh_analysis.py "coh" 1 0 "$1" "cue" 1
python -O mi_coh_analysis.py "coh" 0 0 "$1" "cue" 0
