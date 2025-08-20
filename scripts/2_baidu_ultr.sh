#!/bin/sh
#SBATCH --job-name=counter
#SBATCH --time=32:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task 16
#SBATCH --mem 256GB
#SBATCH --partition cpu
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=p.k.hager@uva.nl

export PYTHONUNBUFFERED=TRUE
python count.py
