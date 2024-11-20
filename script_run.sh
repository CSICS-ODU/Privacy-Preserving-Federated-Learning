#!/bin/bash

#SBATCH -c 8
#SBATCH -p gpu --gres=gpu:1
#SBATCH -o new-Cifar100-2-%j.out
module load container_env pytorch-gpu

crun -p /scratch/sbane002/shared/envs/new_nvflare python3 fedavg_script_runner_pt.py -n 2 -r 2 -d CIFAR100
