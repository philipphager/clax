#!/usr/bin/env bash

python main.py -m \
  experiment=1-yandex-10m/1/ \
  model='glob(*)' \
  dataset=yandex \
  query_doc_pairs=61_600_000 \
  train_sessions=[0,10_000_000] \
  val_sessions=[10_000_000,15_000_000] \
  test_sessions=[15_000_000,20_000_000] \
  eval_train_queries_only=True \
  parameter/attraction=embedding/full \
  parameter/satisfaction=embedding/full \
  random_state=1 \
  $@

python main.py -m \
  experiment=1-yandex-10m/2/ \
  model='glob(*)' \
  dataset=yandex \
  query_doc_pairs=61_600_000 \
  train_sessions=[10_000_000,20_000_000] \
  val_sessions=[20_000_000,25_000_000] \
  test_sessions=[25_000_000,30_000_000] \
  eval_train_queries_only=True \
  parameter/attraction=embedding/full \
  parameter/satisfaction=embedding/full \
  random_state=2 \
  $@

python main.py -m \
  experiment=1-yandex-10m/3/ \
  model='glob(*)' \
  dataset=yandex \
  query_doc_pairs=61_600_000 \
  train_sessions=[20_000_000,30_000_000] \
  val_sessions=[30_000_000,35_000_000] \
  test_sessions=[35_000_000,40_000_000] \
  eval_train_queries_only=True \
  parameter/attraction=embedding/full \
  parameter/satisfaction=embedding/full \
  random_state=3 \
  $@
