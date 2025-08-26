#!/usr/bin/env bash
#SBATCH --job-name=2-yandex-compression
#SBATCH --partition=cpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=4GB
#SBATCH --time=24:00:00

python main.py -m \
  experiment=2-yandex-compression/full/1 \
  model=gctr,rctr,dctr,pbm,ubm,cm,ccm,dbn,sdbn,dcm \
  parameter/attraction=embedding/full \
  parameter/satisfaction=embedding/full \
  train_sessions=[0,80_000_000] \
  val_sessions=[80_000_000,90_000_000] \
  test_sessions=[90_000_000,100_000_000] \
  random_state=1 \
  $@

python main.py -m \
  experiment=2-yandex-compression/full/2 \
  model=gctr,rctr,dctr,pbm,ubm,cm,ccm,dbn,sdbn,dcm \
  parameter/attraction=embedding/full \
  parameter/satisfaction=embedding/full \
  train_sessions=[20_000_000,100_000_000] \
  val_sessions=[100_000_000,110_000_000] \
  test_sessions=[110_000_000,120_000_000] \
  random_state=2 \
  $@

python main.py -m \
  experiment=2-yandex-compression/full/3 \
  model=gctr,rctr,dctr,pbm,ubm,cm,ccm,dbn,sdbn,dcm \
  parameter/attraction=embedding/full \
  parameter/satisfaction=embedding/full \
  train_sessions=[40_000_000,120_000_000] \
  val_sessions=[120_000_000,130_000_000] \
  test_sessions=[130_000_000,140_000_000] \
  random_state=3 \
  $@

python main.py -m \
  experiment=2-yandex-compression/hash/2/1 \
  model=gctr,rctr,dctr,pbm,ubm,cm,ccm,dbn,sdbn,dcm \
  parameter/attraction=embedding/hash \
  parameter/satisfaction=embedding/hash \
  compression_ratio=2 \
  train_sessions=[0,80_000_000] \
  val_sessions=[80_000_000,90_000_000] \
  test_sessions=[90_000_000,100_000_000] \
  random_state=1 \
  $@

python main.py -m \
  experiment=2-yandex-compression/hash/2/2 \
  model=gctr,rctr,dctr,pbm,ubm,cm,ccm,dbn,sdbn,dcm \
  parameter/attraction=embedding/hash \
  parameter/satisfaction=embedding/hash \
  compression_ratio=2 \
  train_sessions=[20_000_000,100_000_000] \
  val_sessions=[100_000_000,110_000_000] \
  test_sessions=[110_000_000,120_000_000] \
  random_state=2 \
  $@

python main.py -m \
  experiment=2-yandex-compression/hash/2/3 \
  model=gctr,rctr,dctr,pbm,ubm,cm,ccm,dbn,sdbn,dcm \
  parameter/attraction=embedding/hash \
  parameter/satisfaction=embedding/hash \
  compression_ratio=2 \
  train_sessions=[40_000_000,120_000_000] \
  val_sessions=[120_000_000,130_000_000] \
  test_sessions=[130_000_000,140_000_000] \
  random_state=3 \
  $@

python main.py -m \
  experiment=2-yandex-compression/hash/5/1 \
  model=gctr,rctr,dctr,pbm,ubm,cm,ccm,dbn,sdbn,dcm \
  parameter/attraction=embedding/hash \
  parameter/satisfaction=embedding/hash \
  compression_ratio=5 \
  train_sessions=[0,80_000_000] \
  val_sessions=[80_000_000,90_000_000] \
  test_sessions=[90_000_000,100_000_000] \
  random_state=1 \
  $@

python main.py -m \
  experiment=2-yandex-compression/hash/5/2 \
  model=gctr,rctr,dctr,pbm,ubm,cm,ccm,dbn,sdbn,dcm \
  parameter/attraction=embedding/hash \
  parameter/satisfaction=embedding/hash \
  compression_ratio=5 \
  train_sessions=[20_000_000,100_000_000] \
  val_sessions=[100_000_000,110_000_000] \
  test_sessions=[110_000_000,120_000_000] \
  random_state=2 \
  $@

python main.py -m \
  experiment=2-yandex-compression/hash/5/3 \
  model=gctr,rctr,dctr,pbm,ubm,cm,ccm,dbn,sdbn,dcm \
  parameter/attraction=embedding/hash \
  parameter/satisfaction=embedding/hash \
  compression_ratio=5 \
  train_sessions=[40_000_000,120_000_000] \
  val_sessions=[120_000_000,130_000_000] \
  test_sessions=[130_000_000,140_000_000] \
  random_state=3 \
  $@

python main.py -m \
  experiment=2-yandex-compression/hash/10/1 \
  model=gctr,rctr,dctr,pbm,ubm,cm,ccm,dbn,sdbn,dcm \
  parameter/attraction=embedding/hash \
  parameter/satisfaction=embedding/hash \
  compression_ratio=10 \
  train_sessions=[0,80_000_000] \
  val_sessions=[80_000_000,90_000_000] \
  test_sessions=[90_000_000,100_000_000] \
  random_state=1 \
  $@

python main.py -m \
  experiment=2-yandex-compression/hash/10/2 \
  model=gctr,rctr,dctr,pbm,ubm,cm,ccm,dbn,sdbn,dcm \
  parameter/attraction=embedding/hash \
  parameter/satisfaction=embedding/hash \
  compression_ratio=10 \
  train_sessions=[20_000_000,100_000_000] \
  val_sessions=[100_000_000,110_000_000] \
  test_sessions=[110_000_000,120_000_000] \
  random_state=2 \
  $@

python main.py -m \
  experiment=2-yandex-compression/hash/10/3 \
  model=gctr,rctr,dctr,pbm,ubm,cm,ccm,dbn,sdbn,dcm \
  parameter/attraction=embedding/hash \
  parameter/satisfaction=embedding/hash \
  compression_ratio=10 \
  train_sessions=[40_000_000,120_000_000] \
  val_sessions=[120_000_000,130_000_000] \
  test_sessions=[130_000_000,140_000_000] \
  random_state=3 \
  $@

python main.py -m \
  experiment=2-yandex-compression/hash/100/1 \
  model=gctr,rctr,dctr,pbm,ubm,cm,ccm,dbn,sdbn,dcm \
  parameter/attraction=embedding/hash \
  parameter/satisfaction=embedding/hash \
  compression_ratio=100 \
  train_sessions=[0,80_000_000] \
  val_sessions=[80_000_000,90_000_000] \
  test_sessions=[90_000_000,100_000_000] \
  random_state=1 \
  $@

python main.py -m \
  experiment=2-yandex-compression/hash/100/2 \
  model=gctr,rctr,dctr,pbm,ubm,cm,ccm,dbn,sdbn,dcm \
  parameter/attraction=embedding/hash \
  parameter/satisfaction=embedding/hash \
  compression_ratio=100 \
  train_sessions=[20_000_000,100_000_000] \
  val_sessions=[100_000_000,110_000_000] \
  test_sessions=[110_000_000,120_000_000] \
  random_state=2 \
  $@

python main.py -m \
  experiment=2-yandex-compression/hash/100/3 \
  model=gctr,rctr,dctr,pbm,ubm,cm,ccm,dbn,sdbn,dcm \
  parameter/attraction=embedding/hash \
  parameter/satisfaction=embedding/hash \
  compression_ratio=100 \
  train_sessions=[40_000_000,120_000_000] \
  val_sessions=[120_000_000,130_000_000] \
  test_sessions=[130_000_000,140_000_000] \
  random_state=3 \
  $@

python main.py -m \
  experiment=2-yandex-compression/hash/1000/1 \
  model=gctr,rctr,dctr,pbm,ubm,cm,ccm,dbn,sdbn,dcm \
  parameter/attraction=embedding/hash \
  parameter/satisfaction=embedding/hash \
  compression_ratio=1000 \
  train_sessions=[0,80_000_000] \
  val_sessions=[80_000_000,90_000_000] \
  test_sessions=[90_000_000,100_000_000] \
  random_state=1 \
  $@

python main.py -m \
  experiment=2-yandex-compression/hash/1000/2 \
  model=gctr,rctr,dctr,pbm,ubm,cm,ccm,dbn,sdbn,dcm \
  parameter/attraction=embedding/hash \
  parameter/satisfaction=embedding/hash \
  compression_ratio=1000 \
  train_sessions=[20_000_000,100_000_000] \
  val_sessions=[100_000_000,110_000_000] \
  test_sessions=[110_000_000,120_000_000] \
  random_state=2 \
  $@

python main.py -m \
  experiment=2-yandex-compression/hash/1000/3 \
  model=gctr,rctr,dctr,pbm,ubm,cm,ccm,dbn,sdbn,dcm \
  parameter/attraction=embedding/hash \
  parameter/satisfaction=embedding/hash \
  compression_ratio=1000 \
  train_sessions=[40_000_000,120_000_000] \
  val_sessions=[120_000_000,130_000_000] \
  test_sessions=[130_000_000,140_000_000] \
  random_state=3 \
  $@

python main.py -m \
  experiment=2-yandex-compression/qr/2/1 \
  model=gctr,rctr,dctr,pbm,ubm,cm,ccm,dbn,sdbn,dcm \
  parameter/attraction=embedding/qr \
  parameter/satisfaction=embedding/qr \
  compression_ratio=2 \
  train_sessions=[0,80_000_000] \
  val_sessions=[80_000_000,90_000_000] \
  test_sessions=[90_000_000,100_000_000] \
  random_state=1 \
  $@

python main.py -m \
  experiment=2-yandex-compression/qr/2/2 \
  model=gctr,rctr,dctr,pbm,ubm,cm,ccm,dbn,sdbn,dcm \
  parameter/attraction=embedding/qr \
  parameter/satisfaction=embedding/qr \
  compression_ratio=2 \
  train_sessions=[20_000_000,100_000_000] \
  val_sessions=[100_000_000,110_000_000] \
  test_sessions=[110_000_000,120_000_000] \
  random_state=2 \
  $@

python main.py -m \
  experiment=2-yandex-compression/qr/2/3 \
  model=gctr,rctr,dctr,pbm,ubm,cm,ccm,dbn,sdbn,dcm \
  parameter/attraction=embedding/qr \
  parameter/satisfaction=embedding/qr \
  compression_ratio=2 \
  train_sessions=[40_000_000,120_000_000] \
  val_sessions=[120_000_000,130_000_000] \
  test_sessions=[130_000_000,140_000_000] \
  random_state=3 \
  $@

python main.py -m \
  experiment=2-yandex-compression/qr/5/1 \
  model=gctr,rctr,dctr,pbm,ubm,cm,ccm,dbn,sdbn,dcm \
  parameter/attraction=embedding/qr \
  parameter/satisfaction=embedding/qr \
  compression_ratio=5 \
  train_sessions=[0,80_000_000] \
  val_sessions=[80_000_000,90_000_000] \
  test_sessions=[90_000_000,100_000_000] \
  random_state=1 \
  $@

python main.py -m \
  experiment=2-yandex-compression/qr/5/2 \
  model=gctr,rctr,dctr,pbm,ubm,cm,ccm,dbn,sdbn,dcm \
  parameter/attraction=embedding/qr \
  parameter/satisfaction=embedding/qr \
  compression_ratio=5 \
  train_sessions=[20_000_000,100_000_000] \
  val_sessions=[100_000_000,110_000_000] \
  test_sessions=[110_000_000,120_000_000] \
  random_state=2 \
  $@

python main.py -m \
  experiment=2-yandex-compression/qr/5/3 \
  model=gctr,rctr,dctr,pbm,ubm,cm,ccm,dbn,sdbn,dcm \
  parameter/attraction=embedding/qr \
  parameter/satisfaction=embedding/qr \
  compression_ratio=5 \
  train_sessions=[40_000_000,120_000_000] \
  val_sessions=[120_000_000,130_000_000] \
  test_sessions=[130_000_000,140_000_000] \
  random_state=3 \
  $@

python main.py -m \
  experiment=2-yandex-compression/qr/10/1 \
  model=gctr,rctr,dctr,pbm,ubm,cm,ccm,dbn,sdbn,dcm \
  parameter/attraction=embedding/qr \
  parameter/satisfaction=embedding/qr \
  compression_ratio=10 \
  train_sessions=[0,80_000_000] \
  val_sessions=[80_000_000,90_000_000] \
  test_sessions=[90_000_000,100_000_000] \
  random_state=1 \
  $@

python main.py -m \
  experiment=2-yandex-compression/qr/10/2 \
  model=gctr,rctr,dctr,pbm,ubm,cm,ccm,dbn,sdbn,dcm \
  parameter/attraction=embedding/qr \
  parameter/satisfaction=embedding/qr \
  compression_ratio=10 \
  train_sessions=[20_000_000,100_000_000] \
  val_sessions=[100_000_000,110_000_000] \
  test_sessions=[110_000_000,120_000_000] \
  random_state=2 \
  $@

python main.py -m \
  experiment=2-yandex-compression/qr/10/3 \
  model=gctr,rctr,dctr,pbm,ubm,cm,ccm,dbn,sdbn,dcm \
  parameter/attraction=embedding/qr \
  parameter/satisfaction=embedding/qr \
  compression_ratio=10 \
  train_sessions=[40_000_000,120_000_000] \
  val_sessions=[120_000_000,130_000_000] \
  test_sessions=[130_000_000,140_000_000] \
  random_state=3 \
  $@

python main.py -m \
  experiment=2-yandex-compression/qr/100/1 \
  model=gctr,rctr,dctr,pbm,ubm,cm,ccm,dbn,sdbn,dcm \
  parameter/attraction=embedding/qr \
  parameter/satisfaction=embedding/qr \
  compression_ratio=100 \
  train_sessions=[0,80_000_000] \
  val_sessions=[80_000_000,90_000_000] \
  test_sessions=[90_000_000,100_000_000] \
  random_state=1 \
  $@

python main.py -m \
  experiment=2-yandex-compression/qr/100/2 \
  model=gctr,rctr,dctr,pbm,ubm,cm,ccm,dbn,sdbn,dcm \
  parameter/attraction=embedding/qr \
  parameter/satisfaction=embedding/qr \
  compression_ratio=100 \
  train_sessions=[20_000_000,100_000_000] \
  val_sessions=[100_000_000,110_000_000] \
  test_sessions=[110_000_000,120_000_000] \
  random_state=2 \
  $@

python main.py -m \
  experiment=2-yandex-compression/qr/100/3 \
  model=gctr,rctr,dctr,pbm,ubm,cm,ccm,dbn,sdbn,dcm \
  parameter/attraction=embedding/qr \
  parameter/satisfaction=embedding/qr \
  compression_ratio=100 \
  train_sessions=[40_000_000,120_000_000] \
  val_sessions=[120_000_000,130_000_000] \
  test_sessions=[130_000_000,140_000_000] \
  random_state=3 \
  $@

python main.py -m \
  experiment=2-yandex-compression/qr/1000/1 \
  model=gctr,rctr,dctr,pbm,ubm,cm,ccm,dbn,sdbn,dcm \
  parameter/attraction=embedding/qr \
  parameter/satisfaction=embedding/qr \
  compression_ratio=1000 \
  train_sessions=[0,80_000_000] \
  val_sessions=[80_000_000,90_000_000] \
  test_sessions=[90_000_000,100_000_000] \
  random_state=1 \
  $@

python main.py -m \
  experiment=2-yandex-compression/qr/1000/2 \
  model=gctr,rctr,dctr,pbm,ubm,cm,ccm,dbn,sdbn,dcm \
  parameter/attraction=embedding/qr \
  parameter/satisfaction=embedding/qr \
  compression_ratio=1000 \
  train_sessions=[20_000_000,100_000_000] \
  val_sessions=[100_000_000,110_000_000] \
  test_sessions=[110_000_000,120_000_000] \
  random_state=2 \
  $@

python main.py -m \
  experiment=2-yandex-compression/qr/1000/3 \
  model=gctr,rctr,dctr,pbm,ubm,cm,ccm,dbn,sdbn,dcm \
  parameter/attraction=embedding/qr \
  parameter/satisfaction=embedding/qr \
  compression_ratio=1000 \
  train_sessions=[40_000_000,120_000_000] \
  val_sessions=[120_000_000,130_000_000] \
  test_sessions=[130_000_000,140_000_000] \
  random_state=3 \
  $@
