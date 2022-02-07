#!/bin/bash

#SBATCH -J MI                # Job name
#SBATCH -o .out/mi_coh_%a.out   # Name of stdout output file (%j expands to %jobID)
#SBATCH --ntasks=40
#SBATCH --mem-per-cpu=2000
#SBATCH --cpus-per-task=1

#python -O mi_net_analysis.py "coh" "degree" 0 
#python -O mi_net_analysis.py "coh" "coreness" 0
#python -O mi_net_analysis.py "coh" "efficiency" 0

#python -O mi_net_analysis.py "plv" "degree" 1 
#python -O mi_net_analysis.py "plv" "coreness" 1
#python -O mi_net_analysis.py "plv" "efficiency" 1

python -O mi_coh_analysis.py "plv" 1

