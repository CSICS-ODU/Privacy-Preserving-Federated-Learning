#!/bin/bash

#SBATCH -c 8
#SBATCH -p gpu 
#SBATCH --gres gpu:1
#SBATCH -o attack_OverlapFL_exp1_CIFAR10_10_fedavg_net-%j.out

module load container_env  pytorch-gpu

crun -p /scratch/sbane002/shared/envs/new_nvflare python3 mi_attack.py -d CIFAR10 -em OverlapFL_exp1_CIFAR10_10_fedavg_net
