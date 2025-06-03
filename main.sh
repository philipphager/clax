#!/bin/bash -l

#SBATCH --time=6:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64GB
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

source ${HOME}/.bashrc
mamba activate two-tower-confounding

python main.py
