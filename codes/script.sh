#!/bin/bash -l

#$ -P myproject       # Specify the SCC project name you want to use
#$ -l h_rt=12:00:00   # Specify the hard time limit for the job
#$ -N myjob           # Give job a name
#$ -j y               # Merge the error and output streams into a single file


source /projectnb/sw-prop/venvs/mynewenv/bin/activate
python /projectnb/sw-prop/github/lexi_term/codes/LEXI_exposure_map_parallel.py