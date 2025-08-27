#!/usr/bin/env bash
#SBATCH --job-name=4-baidu-ultr-features
#SBATCH --partition=cpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=4GB
#SBATCH --time=24:00:00

uv run main.py -m \
  experiment=4-baidu-ultr-features/1/ \
  model=gctr,rctr,dctr,pbm,ubm,cm,ccm,dbn,sdbn,dcm \
  train_sessions=[0,1_850_000] \
  val_sessions=[1_850_000,2_100_000] \
  test_sessions=[2_100_000,2_350_000] \
  random_state=1 \
  $@

uv run main.py -m \
  experiment=4-baidu-ultr-features/2/ \
  model=gctr,rctr,dctr,pbm,ubm,cm,ccm,dbn,sdbn,dcm \
  train_sessions=[250_000,2_100_000] \
  val_sessions=[2_100_000,2_350_000] \
  test_sessions=[0,250_000] \
  random_state=2 \
  $@

uv run main.py -m \
  experiment=4-baidu-ultr-features/3/ \
  model=gctr,rctr,dctr,pbm,ubm,cm,ccm,dbn,sdbn,dcm \
  train_sessions=[500_000,2_350_000] \
  val_sessions=[0,250_000] \
  test_sessions=[250_000,500_000] \
  random_state=3 \
  $@
