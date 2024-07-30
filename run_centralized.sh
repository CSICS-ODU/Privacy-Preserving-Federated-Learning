#!/bin/bash

#SBATCH -c 8
#SBATCH -p gpu 
#SBATCH --gres gpu:1
#SBATCH -o centralizedincrementalSVHN-ABCD-%j.out

module load container_env  pytorch-gpu/2.0.1

crun -p ~/envs/ppfl python code/centralized.py -e 20 -m efficientnet -d incrementalSVHN=ABCD_4 -no_bar #-w  add rest of your arguments