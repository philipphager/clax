from pathlib import Path
from time import perf_counter

import optax
import pandas as pd
from flax import nnx
from torch.utils.data import DataLoader
from jax.lib import xla_bridge

from clix.datasets.yandex import YandexDataset
from clix.models import (
    GlobalCTRModel,
    DocumentBasedCTRModel,
    RankBasedCTRModel,
    UserBrowsingModel,
    DynamicBayesianNetwork,
    ClickChainModel,
    DependentClickModel,
)
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
    print(xla_bridge.get_backend().platform)

    rngs = nnx.Rngs(0)
    path = Path("data/wscd-2012/YandexClicks.txt")
    index_path = Path("data/wscd-2012/index.json")

    train_dataset = YandexDataset(path, index_path, session_range=(0, 6_000_000))
    val_dataset = YandexDataset(path, index_path, session_range=(6_000_000, 8_000_000))
    test_dataset = YandexDataset(
        path, index_path, session_range=(8_000_000, 10_000_000)
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
        num_workers=4,
        collate_fn=test_dataset.collate_fn,
    )

    query_doc_pairs = 10_000_000
    positions = 10

    models = [
        # GlobalCTRModel(rngs=rngs),
        DocumentBasedCTRModel(rngs=rngs, query_doc_pairs=query_doc_pairs),
        # RankBasedCTRModel(rngs=rngs, positions=positions),
        # PositionBasedModel(
        #     rngs=rngs,
        #     query_doc_pairs=query_doc_pairs,
        #     positions=positions,
        # ),
        # UserBrowsingModel(
        #     rngs=rngs,
        #     query_doc_pairs=query_doc_pairs,
        #     positions=positions,
        # ),
        # DynamicBayesianNetwork(
        #     rngs=rngs,
        #     query_doc_pairs=query_doc_pairs,
        # ),
        # ClickChainModel(
        #     rngs=rngs,
        #     query_doc_pairs=query_doc_pairs,
        # ),
        # DependentClickModel(
        #     rngs=rngs,
        #     query_doc_pairs=query_doc_pairs,
        #     positions=positions,
        # ),
    ]

    train_dfs = []
    test_dfs = []

    for model in models:
        trainer = Trainer(optax.adamw(1e-3), epochs=50, patience=0)

        timer_start = perf_counter()
        train_df = trainer.train(model, train_loader, val_loader)
        timer_stop = perf_counter()
        train_df["train_time_s"] = timer_stop - timer_start
        train_dfs.append(train_df)

        test_df = trainer.test(model, test_loader)
        test_dfs.append(test_df)

        pd.concat(train_dfs).to_csv("train.csv")
        pd.concat(test_dfs).to_csv("test.csv")


if __name__ == "__main__":
    main()
