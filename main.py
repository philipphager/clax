from pathlib import Path

import optax
import torch
from flax import nnx
from torch.utils.data import DataLoader

from clix.models import (
    PositionBasedModel,
)
from clix.trainer import Trainer
from clix.yandex import YandexDataset


def main():
    rngs = nnx.Rngs(0)
    import jax

    print(jax.devices())

    path = Path("/ivi/ilps/datasets/yandex/relevance_prediction/YandexClicks.txt")
    index_path = Path("data/wscd-2012/index.json")

    train_dataset = YandexDataset(path, index_path, session_range=(0, 10_000_000))
    val_dataset = YandexDataset(
        path, index_path, session_range=(10_000_000, 15_000_000)
    )
    test_dataset = YandexDataset(
        path, index_path, session_range=(15_000_000, 20_000_000)
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=512,
        num_workers=1,
        collate_fn=train_dataset.collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=512,
        num_workers=1,
        collate_fn=val_dataset.collate_fn,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=512,
        num_workers=1,
        collate_fn=test_dataset.collate_fn,
    )
    model = PositionBasedModel(rngs=rngs, query_doc_pairs=120_000_000, positions=10)
    trainer = Trainer(optax.adam(1e-3), epochs=10, patience=0)
    train_df = trainer.train(model, train_loader, val_loader)
    test_df = trainer.test(model, test_loader)

    train_df.to_csv("train.csv")
    test_df.to_csv("test.csv")


if __name__ == "__main__":
    main()
