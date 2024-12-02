#!/bin/bash

#SBATCH -c 8
#SBATCH -p gpu --gres=gpu:1
#SBATCH -o scaffold-Cifar10-10-%j.out
module load container_env pytorch-gpu

crun -p /scratch/sbane002/shared/envs/new_nvflare python3 scaffold_script_runner_pt.py -n 10 -r 2 -d CIFAR10 -a scaffold
