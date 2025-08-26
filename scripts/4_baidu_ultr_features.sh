#!/usr/bin/env bash
#SBATCH --job-name=baidu-ultr-features
#SBATCH --partition=cpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=4GB
#SBATCH --time=24:00:00

python main.py -m \
  experiment=4-baidu-ultr-features/1/ \
  model='glob(*)' \
  dataset=baidu-ultr-uva \
  parameter/attraction=deep-cross \
  parameter/satisfaction=deep-cross \
  train_sessions=[0,1_750_000] \
  val_sessions=[1_750_000,2_000_000] \
  test_sessions=[2_000_000,2_250_000] \
  random_state=1 \
  $@

python main.py -m \
  experiment=4-baidu-ultr-features/2/ \
  model='glob(*)' \
  dataset=baidu-ultr-uva \
  parameter/attraction=deep-cross \
  parameter/satisfaction=deep-cross \
  train_sessions=[250_000,2_000_000] \
  val_sessions=[2_000_000,2_250_000] \
  test_sessions=[0,250_000] \
  random_state=2 \
  $@

python main.py -m \
  experiment=4-baidu-ultr-features/3/ \
  model='glob(*)' \
  dataset=baidu-ultr-uva \
  parameter/attraction=deep-cross \
  parameter/satisfaction=deep-cross \
  train_sessions=[500_000,2_250_000] \
  val_sessions=[0,250_000] \
  test_sessions=[250_000,500_000] \
  random_state=3 \
  $@
