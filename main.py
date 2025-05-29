import optax
import torch
from flax import nnx
from torch.utils.data import DataLoader

from clix.datasets import YandexDataset
from clix.models import (
    PositionBasedModel,
)
from clix.trainer import Trainer


def main():
    rngs = nnx.Rngs(0)
    dataset = YandexDataset("data/yandex.csv")
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [0.6, 0.4])
    val_dataset, test_dataset = torch.utils.data.random_split(val_dataset, [0.5, 0.5])

    train_loader = DataLoader(
        train_dataset,
        batch_size=512,
        collate_fn=dataset.collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=512,
        collate_fn=dataset.collate_fn,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=512,
        collate_fn=dataset.collate_fn,
    )
    model = PositionBasedModel(rngs=rngs, query_doc_pairs=140_000, positions=10)
    trainer = Trainer(optax.adam(1e-3), epochs=10, patience=0)
    train_df = trainer.train(model, train_loader, val_loader)
    test_df = trainer.test(model, test_loader)

    train_df.to_csv("train.csv")
    test_df.to_csv("test.csv")


if __name__ == "__main__":
    main()
