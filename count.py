from pathlib import Path
from time import perf_counter

import hydra
import optax
from flax import nnx
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from clax.models import (
    UserBrowsingModel,
    DynamicBayesianNetwork,
    MixtureModel,
)
from clax.trainer import Trainer


@hydra.main(version_base="1.3", config_path="clax/config/", config_name="config")
def main(config: DictConfig):
    print(OmegaConf.to_yaml(config))
    rngs = nnx.Rngs(config.random_state)

    filter_query_ids = None
    train_dataset = instantiate(config.dataset, session_range=config.train_sessions)

    if config.min_train_sessions_per_eval_query > 0:
        filter_query_ids = train_dataset.unique_query_ids(
            min_sessions=config.min_train_sessions_per_eval_query
        )
        print(
            f"Filtering val/test datasets to {len(filter_query_ids):_} unique train "
            f"queries with at least {config.min_train_sessions_per_eval_query} sessions."
        )

    val_dataset = instantiate(
        config.dataset,
        session_range=config.val_sessions,
        filter_query_ids=filter_query_ids,
    )
    test_dataset = instantiate(
        config.dataset,
        session_range=config.test_sessions,
        filter_query_ids=filter_query_ids,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.train_batch_size,
        collate_fn=train_dataset.collate_fn,
        num_workers=8,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.eval_batch_size,
        collate_fn=val_dataset.collate_fn,
        num_workers=8,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.eval_batch_size,
        collate_fn=test_dataset.collate_fn,
        num_workers=8,
    )

    unique_query_doc_ids = set()
    i = 0

    for batch in train_loader:
        if i % 10_000_000 == 0:
            print(i, len(unique_query_doc_ids))

        unique_query_doc_ids.update(set(batch["query_doc_ids"].ravel()))
        i += 1

    for batch in val_loader:
        if i % 10_000_000 == 0:
            print(i, len(unique_query_doc_ids))

        unique_query_doc_ids.update(set(batch["query_doc_ids"].ravel()))
        i += 1

    for batch in test_loader:
        if i % 10_000_000 == 0:
            print(i, len(unique_query_doc_ids))

        unique_query_doc_ids.update(set(batch["query_doc_ids"].ravel()))
        i += 1


if __name__ == "__main__":
    main()
