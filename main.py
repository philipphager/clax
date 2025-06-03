from pathlib import Path

import optax
import pandas as pd
from flax import nnx
from torch.utils.data import DataLoader

from clix.datasets.yandex import YandexDataset
from clix.models.pbm import PositionBasedModel
from clix.trainer import Trainer


def get_judged_queries(path: Path):
    """
    Data Format Source:
    https://web.archive.org/web/20130828200420/http://imat-relpred.yandex.ru/en/datasets
    """
    test_df = pd.read_csv(
        path,
        sep="\t",
        names=["query_id", "region_id", "query_doc_ids", "label"],
        usecols=["query_id"],
    )
    return set(test_df["query_id"].unique().tolist())


def main():
    rngs = nnx.Rngs(0)
    path = Path("data/wscd-2012/YandexClicks.txt")
    index_path = Path("data/wscd-2012/index.json")

    train_dataset = YandexDataset(
        path,
        index_path,
        session_range=(0, 600_000),
    )
    val_dataset = YandexDataset(
        path,
        index_path,
        session_range=(600_000, 800_000),
    )
    test_dataset = YandexDataset(
        path,
        index_path,
        session_range=(800_000, 1_000_000),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=512,
        num_workers=8,
        collate_fn=train_dataset.collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=512,
        num_workers=8,
        collate_fn=val_dataset.collate_fn,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=512,
        num_workers=1,
        collate_fn=test_dataset.collate_fn,
    )

    model = PositionBasedModel(rngs=rngs, query_doc_pairs=5_000_000, positions=10)
    trainer = Trainer(optax.adamw(1e-3), epochs=10, patience=0)
    train_df = trainer.train(model, train_loader, val_loader)
    test_df = trainer.test(model, test_loader)

    train_df.to_csv("train.csv")
    test_df.to_csv("test.csv")


if __name__ == "__main__":
    main()
