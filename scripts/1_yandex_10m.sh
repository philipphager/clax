#!/usr/bin/env bash

python main.py -m \
  experiment=1-yandex-10m/1/ \
  model='glob(*)' \
  dataset=yandex \
  train_sessions=[0,1_000_000] \
  val_sessions=[1_000_000,2_000_000] \
  test_sessions=[2_000_000,3_000_000] \
  min_train_sessions_per_eval_query=10 \
  parameter/attraction=embedding/full \
  parameter/satisfaction=embedding/full \
  random_state=1 \
  $@
#
#python main.py -m \
#  experiment=1-yandex-10m/2/ \
#  model='glob(*)' \
#  dataset=yandex \
#  train_sessions=[10_000_000,20_000_000] \
#  val_sessions=[20_000_000,25_000_000] \
#  test_sessions=[25_000_000,30_000_000] \
#  min_train_sessions_per_eval_query=10 \
#  parameter/attraction=embedding/full \
#  parameter/satisfaction=embedding/full \
#  random_state=2 \
#  $@
#
#python main.py -m \
#  experiment=1-yandex-10m/3/ \
#  model='glob(*)' \
#  dataset=yandex \
#  train_sessions=[20_000_000,30_000_000] \
#  val_sessions=[30_000_000,35_000_000] \
#  test_sessions=[35_000_000,40_000_000] \
#  min_train_sessions_per_eval_query=10 \
#  parameter/attraction=embedding/full \
#  parameter/satisfaction=embedding/full \
#  random_state=3 \
#  $@
