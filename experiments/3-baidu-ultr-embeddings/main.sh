#!/usr/bin/env bash
#SBATCH --job-name=3-baidu-ultr-embeddings
#SBATCH --partition=cpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=4GB
#SBATCH --time=24:00:00

python main.py -m \
  experiment=3-baidu-ultr/1/ \
  model=gctr,rctr,dctr,pbm,ubm,cm,ccm,dbn,sdbn,dcm \
  train_sessions=[0,600_000_000] \
  val_sessions=[600_000_000,700_000_000] \
  test_sessions=[700_000_000,800_000_000] \
  compression_ratio=10 \
  random_state=1 \
  $@

python main.py -m \
  experiment=3-baidu-ultr/2/ \
  model=gctr,rctr,dctr,pbm,ubm,cm,ccm,dbn,sdbn,dcm \
  dataset=baidu-ultr \
  train_sessions=[200_000_000,800_000_000] \
  val_sessions=[800_000_000,900_000_000] \
  test_sessions=[900_000_000,1_000_000_000] \
  random_state=2 \
  $@

python main.py -m \
  experiment=3-baidu-ultr/3/ \
  model=gctr,rctr,dctr,pbm,ubm,cm,ccm,dbn,sdbn,dcm \
  train_sessions=[400_000_000,1_000_000_000] \
  val_sessions=[1_000_000_000,1_100_000_000] \
  test_sessions=[1_100_000_000,1_200_000_000] \
  random_state=3 \
  $@
