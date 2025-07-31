#!/usr/bin/env bash

python main.py -m \
  experiment=1-yandex-10m \
  model='glob(*)' \
  dataset=yandex \
  train_sessions=[0,10_000_000] \
  val_sessions=[10_000_000,15_000_000] \
  test_sessions=[15_000_000,20_000_000] \
  eval_train_queries_only=True \
  parameter/attraction=embedding/full \
  parameter/satisfaction=embedding/full \
  random_state=2025 \
  $@
