#!/usr/bin/env bash
#
#python main.py -m \
#  experiment=2-yandex-embeddings/full/ \
#  model='glob(*)' \
#  dataset=yandex \
#  parameter/attraction=embedding/full \
#  parameter/satisfaction=embedding/full \
#  random_state=1 \
#  $@
#
#python main.py -m \
#  experiment=2-yandex-embeddings/hash/10 \
#  model='glob(*)' \
#  dataset=yandex \
#  parameter/attraction=embedding/hash \
#  parameter/satisfaction=embedding/hash \
#  compression_ratio=10 \
#  random_state=1 \
#  $@
#
#python main.py -m \
#  experiment=2-yandex-embeddings/hash/100 \
#  model='glob(*)' \
#  dataset=yandex \
#  parameter/attraction=embedding/hash \
#  parameter/satisfaction=embedding/hash \
#  compression_ratio=100 \
#  random_state=1 \
#  $@
#
#python main.py -m \
#  experiment=2-yandex-embeddings/hash/1000 \
#  model='glob(*)' \
#  dataset=yandex \
#  parameter/attraction=embedding/hash \
#  parameter/satisfaction=embedding/hash \
#  compression_ratio=1000 \
#  random_state=1 \
#  $@

python main.py -m \
  experiment=2-yandex-embeddings/qr/10 \
  model='glob(*)' \
  dataset=yandex \
  parameter/attraction=embedding/qr \
  parameter/satisfaction=embedding/qr \
  compression_ratio=10 \
  random_state=1 \
  $@

python main.py -m \
  experiment=2-yandex-embeddings/qr/100 \
  model='glob(*)' \
  dataset=yandex \
  parameter/attraction=embedding/qr \
  parameter/satisfaction=embedding/qr \
  compression_ratio=100 \
  random_state=1 \
  $@

python main.py -m \
  experiment=2-yandex-embeddings/qr/1000 \
  model='glob(*)' \
  dataset=yandex \
  parameter/attraction=embedding/qr \
  parameter/satisfaction=embedding/qr \
  compression_ratio=1_000 \
  random_state=1 \
  $@
