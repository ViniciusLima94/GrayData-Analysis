#!/bin/bash

#SBATCH -J MI                # Job name
#SBATCH -o .out/net_%a.out   # Name of stdout output file (%j expands to %jobID)
#SBATCH --ntasks=40
#SBATCH --mem-per-cpu=2000
#SBATCH --cpus-per-task=1
#SBATCH --array=0-25


#parser.add_argument("METRIC",
                    #help="which metric to load",
                    #type=str)
#parser.add_argument("SIDX",
                    #help="index of the session to load",
                    #type=int)
#parser.add_argument("ALIGNED", help="wheter to align data to cue or match",
                    #type=str)
#parser.add_argument("MONKEY", help="which monkey to use",
                    #type=str)

python -O save_network_analysis.py "coh" $SLURM_ARRAY_TASK_ID "cue" "ethyl"
