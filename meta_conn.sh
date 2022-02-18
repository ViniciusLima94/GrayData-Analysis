#!/bin/bash

#SBATCH -J MC                # Job name
#SBATCH -o .out/MC_%a.out    # Name of stdout output file (%j expands to %jobID)
#SBATCH --ntasks=40
#SBATCH --mem-per-cpu=2000
#SBATCH --cpus-per-task=1
#SBATCH --array=0-61

python -O meta_conn.py $SLURM_ARRAY_TASK_ID "coh"
python -O meta_conn.py $SLURM_ARRAY_TASK_ID "plv"
python -O meta_conn.py $SLURM_ARRAY_TASK_ID "pec"

