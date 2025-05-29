#!/bin/bash

#SBATCH -c 8
#SBATCH -p gpu --gres=gpu:1
#SBATCH -o OverlapFL_exp5_fedavg-Cifar10-10-%j.out
module load container_env pytorch-gpu/2.2

crun -p /scratch/sbane002/shared/envs/new_nvflare python3 fedavg_script_runner_pt.py -n 10 -r 1 -d CIFAR10 -a fedavg
