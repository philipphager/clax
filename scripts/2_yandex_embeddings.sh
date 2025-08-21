#!/usr/bin/env bash
#SBATCH --job-name=compression
#SBATCH --partition=cpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=4GB
#SBATCH --time=24:00:00

python main.py -m \
  experiment=2-yandex-embeddings/full/1 \
  model='glob(*)' \
  dataset=yandex \
  parameter/attraction=embedding/full \
  parameter/satisfaction=embedding/full \
  random_state=1 \
  $@

python main.py -m \
  experiment=2-yandex-embeddings/full/2 \
  model='glob(*)' \
  dataset=yandex \
  parameter/attraction=embedding/full \
  parameter/satisfaction=embedding/full \
  random_state=2 \
  $@

python main.py -m \
  experiment=2-yandex-embeddings/full/3 \
  model='glob(*)' \
  dataset=yandex \
  parameter/attraction=embedding/full \
  parameter/satisfaction=embedding/full \
  random_state=3 \
  $@

python main.py -m \
  experiment=2-yandex-embeddings/hash/10/1 \
  model='glob(*)' \
  dataset=yandex \
  parameter/attraction=embedding/hash \
  parameter/satisfaction=embedding/hash \
  compression_ratio=10 \
  random_state=1 \
  $@

python main.py -m \
  experiment=2-yandex-embeddings/hash/10/2 \
  model='glob(*)' \
  dataset=yandex \
  parameter/attraction=embedding/hash \
  parameter/satisfaction=embedding/hash \
  compression_ratio=10 \
  random_state=2 \
  $@

python main.py -m \
  experiment=2-yandex-embeddings/hash/10/3 \
  model='glob(*)' \
  dataset=yandex \
  parameter/attraction=embedding/hash \
  parameter/satisfaction=embedding/hash \
  compression_ratio=10 \
  random_state=3 \
  $@

python main.py -m \
  experiment=2-yandex-embeddings/hash/100/1 \
  model='glob(*)' \
  dataset=yandex \
  parameter/attraction=embedding/hash \
  parameter/satisfaction=embedding/hash \
  compression_ratio=100 \
  random_state=1 \
  $@

python main.py -m \
  experiment=2-yandex-embeddings/hash/100/2 \
  model='glob(*)' \
  dataset=yandex \
  parameter/attraction=embedding/hash \
  parameter/satisfaction=embedding/hash \
  compression_ratio=100 \
  random_state=2 \
  $@

python main.py -m \
  experiment=2-yandex-embeddings/hash/100/3 \
  model='glob(*)' \
  dataset=yandex \
  parameter/attraction=embedding/hash \
  parameter/satisfaction=embedding/hash \
  compression_ratio=100 \
  random_state=3 \
  $@

python main.py -m \
  experiment=2-yandex-embeddings/hash/1000/1 \
  model='glob(*)' \
  dataset=yandex \
  parameter/attraction=embedding/hash \
  parameter/satisfaction=embedding/hash \
  compression_ratio=1000 \
  random_state=1 \
  $@

python main.py -m \
  experiment=2-yandex-embeddings/hash/1000/2 \
  model='glob(*)' \
  dataset=yandex \
  parameter/attraction=embedding/hash \
  parameter/satisfaction=embedding/hash \
  compression_ratio=1000 \
  random_state=2 \
  $@

python main.py -m \
  experiment=2-yandex-embeddings/hash/1000/3 \
  model='glob(*)' \
  dataset=yandex \
  parameter/attraction=embedding/hash \
  parameter/satisfaction=embedding/hash \
  compression_ratio=1000 \
  random_state=3 \
  $@



python main.py -m \
  experiment=2-yandex-embeddings/qr/10/1 \
  model='glob(*)' \
  dataset=yandex \
  parameter/attraction=embedding/qr \
  parameter/satisfaction=embedding/qr \
  compression_ratio=10 \
  random_state=1 \
  $@

python main.py -m \
  experiment=2-yandex-embeddings/qr/10/2 \
  model='glob(*)' \
  dataset=yandex \
  parameter/attraction=embedding/qr \
  parameter/satisfaction=embedding/qr \
  compression_ratio=10 \
  random_state=2 \
  $@

python main.py -m \
  experiment=2-yandex-embeddings/qr/10/3 \
  model='glob(*)' \
  dataset=yandex \
  parameter/attraction=embedding/qr \
  parameter/satisfaction=embedding/qr \
  compression_ratio=10 \
  random_state=3 \
  $@

python main.py -m \
  experiment=2-yandex-embeddings/qr/100/1 \
  model='glob(*)' \
  dataset=yandex \
  parameter/attraction=embedding/qr \
  parameter/satisfaction=embedding/qr \
  compression_ratio=100 \
  random_state=1 \
  $@

python main.py -m \
  experiment=2-yandex-embeddings/qr/100/2 \
  model='glob(*)' \
  dataset=yandex \
  parameter/attraction=embedding/qr \
  parameter/satisfaction=embedding/qr \
  compression_ratio=100 \
  random_state=2 \
  $@

python main.py -m \
  experiment=2-yandex-embeddings/qr/100/3 \
  model='glob(*)' \
  dataset=yandex \
  parameter/attraction=embedding/qr \
  parameter/satisfaction=embedding/qr \
  compression_ratio=100 \
  random_state=3 \
  $@

python main.py -m \
  experiment=2-yandex-embeddings/qr/1000/1 \
  model='glob(*)' \
  dataset=yandex \
  parameter/attraction=embedding/qr \
  parameter/satisfaction=embedding/qr \
  compression_ratio=1000 \
  random_state=1 \
  $@

python main.py -m \
  experiment=2-yandex-embeddings/qr/1000/2 \
  model='glob(*)' \
  dataset=yandex \
  parameter/attraction=embedding/qr \
  parameter/satisfaction=embedding/qr \
  compression_ratio=1000 \
  random_state=2 \
  $@

python main.py -m \
  experiment=2-yandex-embeddings/qr/1000/3 \
  model='glob(*)' \
  dataset=yandex \
  parameter/attraction=embedding/qr \
  parameter/satisfaction=embedding/qr \
  compression_ratio=1000 \
  random_state=3 \
  $@
