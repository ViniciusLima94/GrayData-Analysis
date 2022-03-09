#!/bin/bash

#SBATCH -J NLI               # Job name
#SBATCH -o .out/nli_%a.out   # Name of stdout output file (%j expands to %jobID)
#SBATCH --ntasks=40
#SBATCH --mem-per-cpu=2000
#SBATCH --cpus-per-task=1
#SBATCH --array=0-62

python -O nli.py $SLURM_ARRAY_TASK_ID 'coh'
#python -O nli.py $SLURM_ARRAY_TASK_ID 'plv'
#python -O nli.py $SLURM_ARRAY_TASK_ID 'pec'
#Rscript nli.R $SLURM_ARRAY_TASK_ID
