#!/bin/bash

#SBATCH -J MI                # Job name
#SBATCH -o .out/plot_features_%a.out   # Name of stdout output file (%j expands to %jobID)
#SBATCH --ntasks=40
#SBATCH --mem-per-cpu=2000
#SBATCH --cpus-per-task=1

python -O plot_features_flatmap.py "power" "mean" 
python -O plot_features_flatmap.py "power" "cv" 
python -O plot_features_flatmap.py "power" "95p" 

python -O plot_features_flatmap.py "degree" "mean" 
python -O plot_features_flatmap.py "degree" "cv" 
python -O plot_features_flatmap.py "degree" "95p" 

python -O plot_features_flatmap.py "coreness" "mean" 
python -O plot_features_flatmap.py "coreness" "cv" 
python -O plot_features_flatmap.py "coreness" "95p" 
