#!/bin/bash

#SBATCH -J MC                # Job name
#SBATCH -o .out/MC_%a.out    # Name of stdout output file (%j expands to %jobID)
#SBATCH --ntasks=40
#SBATCH --mem-per-cpu=2000
#SBATCH --cpus-per-task=1
#SBATCH --array=4-61

#parser.add_argument("SIDX", help="index of the session to run",
                    #type=int)
#parser.add_argument("METRIC",
                    #help="which FC metric to use",
                    #type=str)
#parser.add_argument("SURR",
                    #help="wheter to use original or surrogate MC",
                    #type=int)
#parser.add_argument("THR",
                    #help="wheter to threshold or not the coherence",
                    #type=int)
#parser.add_argument("MONKEY", help="which monkey to use",
                    #type=str)
#parser.add_argument("ALIGNED", help="wheter power was align to cue or match",
                    #type=str)
#parser.add_argument("DELAY", help="which type of delay split to use",
                    #type=int)


# Original
                       # SIDX           METRIC SURR THR MONKEY
python -O meta_conn.py $SLURM_ARRAY_TASK_ID "coh" 0 0 "lucy" "cue" 1 
python -O meta_conn.py $SLURM_ARRAY_TASK_ID "coh" 0 1 "lucy" "cue" 1
# Surrogate
python -O meta_conn.py $SLURM_ARRAY_TASK_ID "coh" 1 0 "lucy" "cue" 1
# Surrogate strong
python -O meta_conn.py $SLURM_ARRAY_TASK_ID "coh" 2 0 "lucy" "cue" 1
