#!/bin/bash -l

#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64GB
#SBATCH --partition=cpu

source ${HOME}/.bashrc
mamba activate two-tower-confounding

python baselines.py
