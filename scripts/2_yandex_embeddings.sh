#!/usr/bin/env bash

python main.py -m \
  experiment=2-yandex-embeddings/full/1/ \
  model='glob(*)' \
  dataset=yandex \
  parameter/attraction=embedding/full \
  parameter/satisfaction=embedding/full \
  random_state=1 \
  $@

python main.py -m \
  experiment=2-yandex-embeddings/hash/1/ \
  model='glob(*)' \
  dataset=yandex \
  parameter/attraction=embedding/hash \
  parameter/satisfaction=embedding/hash \
  compression_ratio=10,100,1_000
  random_state=1 \
  $@

python main.py -m \
  experiment=2-yandex-embeddings/qr/1/ \
  model='glob(*)' \
  dataset=yandex \
  parameter/attraction=embedding/qr \
  parameter/satisfaction=embedding/qr \
  compression_ratio=10,100,1_000
  random_state=1 \
  $@
