#!/usr/bin/env bash

python main.py -m \
  experiment=3-baidu-ultr/1/ \
  model='glob(*)' \
  dataset=baidu-ultr \
  parameter/attraction=embedding/hash \
  parameter/satisfaction=embedding/hash \
  compression_ratio=1_000 \
  random_state=1 \
  $@

python main.py -m \
  experiment=3-baidu-ultr/2/ \
  model='glob(*)' \
  dataset=baidu-ultr \
  parameter/attraction=embedding/hash \
  parameter/satisfaction=embedding/hash \
  compression_ratio=1_000 \
  random_state=2 \
  $@

python main.py -m \
  experiment=3-baidu-ultr/3/ \
  model='glob(*)' \
  dataset=baidu-ultr \
  parameter/attraction=embedding/hash \
  parameter/satisfaction=embedding/hash \
  compression_ratio=1_000 \
  random_state=3 \
  $@
