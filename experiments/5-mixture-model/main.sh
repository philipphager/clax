#!/usr/bin/env bash
#SBATCH --job-name=5-mixture-model
#SBATCH --partition=cpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=4GB
#SBATCH --time=24:00:00

uv run main.py -m \
  experiment=5-mixture-model/1/ \
  train_sessions=[0,1_850_000] \
  val_sessions=[1_850_000,2_100_000] \
  test_sessions=[2_100_000,2_350_000] \
  random_state=1 \
  $@

uv run main.py -m \
  experiment=5-mixture-model/2/ \
  train_sessions=[250_000,2_100_000] \
  val_sessions=[2_100_000,2_350_000] \
  test_sessions=[0,250_000] \
  random_state=2 \
  $@

uv run main.py -m \
  experiment=5-mixture-model/3/ \
  train_sessions=[500_000,2_350_000] \
  val_sessions=[0,250_000] \
  test_sessions=[250_000,500_000] \
  random_state=3 \
  $@
