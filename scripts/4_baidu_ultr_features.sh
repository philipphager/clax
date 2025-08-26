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
  parameter/attraction=embedding/deep-cross \
  parameter/satisfaction=embedding/deep-cross \
  train_sessions=[0,1_000_000] \
  val_sessions=[1_000_000,1_200_000] \
  test_sessions=[1_200_000,1_400_000] \
  random_state=1 \
  $@

python main.py -m \
  experiment=4-baidu-ultr-features/2/ \
  model='glob(*)' \
  dataset=baidu-ultr-uva \
  parameter/attraction=embedding/deep-cross \
  parameter/satisfaction=embedding/deep-cross \
  train_sessions=[400_000,1_400_000] \
  val_sessions=[1_400_000,1_600_000] \
  test_sessions=[1_600_000,1_800_000] \
  random_state=2 \
  $@

python main.py -m \
  experiment=4-baidu-ultr-features/3/ \
  model='glob(*)' \
  dataset=baidu-ultr-uva \
  parameter/attraction=embedding/deep-cross \
  parameter/satisfaction=embedding/deep-cross \
  train_sessions=[800_000,1_800_000] \
  val_sessions=[1_800_000,2_000_000] \
  test_sessions=[2_000_000,2_200_000] \
  random_state=3 \
  $@
