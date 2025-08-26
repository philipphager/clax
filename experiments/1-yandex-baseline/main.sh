#!/usr/bin/env bash
#SBATCH --job-name=1-yandex-baseline
#SBATCH --partition=cpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=4GB
#SBATCH --time=8:00:00


python main.py -m \
  experiment=1-yandex-10m/1/ \
  model=gctr,rctr,dctr,pbm,ubm,cm,ccm,dbn,sdbn,dcm \
  train_sessions=[0,10_000_000] \
  val_sessions=[10_000_000,15_000_000] \
  test_sessions=[15_000_000,20_000_000] \
  random_state=1 \
  $@

python main.py -m \
  experiment=1-yandex-10m/2/ \
  model=gctr,rctr,dctr,pbm,ubm,cm,ccm,dbn,sdbn,dcm \
  train_sessions=[10_000_000,20_000_000] \
  val_sessions=[20_000_000,25_000_000] \
  test_sessions=[25_000_000,30_000_000] \
  random_state=2 \
  $@

python main.py -m \
  experiment=1-yandex-10m/3/ \
  model=gctr,rctr,dctr,pbm,ubm,cm,ccm,dbn,sdbn,dcm \
  train_sessions=[20_000_000,30_000_000] \
  val_sessions=[30_000_000,35_000_000] \
  test_sessions=[35_000_000,40_000_000] \
  random_state=3 \
  $@
