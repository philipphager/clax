#!/usr/bin/env bash

python main.py -m \
  experiment=1-yandex-10m \
  data=yandex \
  train_sessions=[0,10_000_000] \
  val_sessions=[10_000_000, 15_000_000] \
  test_sessions=[15_000_000, 20_000_000] \
  eval_train_queries_only=True \
  random_state=2025
  $@
