from functools import partial
from pathlib import Path
from typing import Tuple

import optax
from flax import nnx
from flax.training.early_stopping import EarlyStopping
from torch.utils.data import DataLoader

from clax import DynamicBayesianNetwork, ClickChainModel
from clax.datasets import YandexDataset
from clax.parameters import EmbeddingParameterConfig, QREmbedding
from clax.parameters.embeddings.compositional import Combination
from clax.trainer import Trainer


def get_yandex_loader(
    dataset_dir: Path,
    session_range: Tuple[int, int],
):
    dataset = YandexDataset(
        dataset_dir=dataset_dir,
        session_range=session_range,
    )

    return DataLoader(
        dataset,
        batch_size=4_096,
        collate_fn=dataset.collate_fn,
        num_workers=4,
        persistent_workers=True,
    )


def main():
    # Scale to the entire Yandex WSCD-2012 dataset with 346_711_929 query-doc pairs
    dataset_dir = Path("../../clax-datasets/yandex")
    query_doc_pairs = 346_711_929

    train_loader = get_yandex_loader(
        dataset_dir,
        session_range=(0, 100_000_000),
    )
    val_loader = get_yandex_loader(
        dataset_dir,
        session_range=(100_000_000, 120_000_000),
    )
    test_loader = get_yandex_loader(
        dataset_dir,
        session_range=(120_000_000, 145_000_000),
    )

    # Instantiate a CCM with Quotient-Remainder compression to reduce the number
    # of allocated embeddings by a factor of 1000x and multiplicative combination:
    rngs = nnx.Rngs(42)

    model = ClickChainModel(
        attraction=EmbeddingParameterConfig(
            use_feature="query_doc_ids",
            embedding_fn=partial(
                QREmbedding,  # Use HashEmbedding for hashing-trick compression
                compression_ratio=1000,
            ),
            parameters=query_doc_pairs,
            add_baseline=True,
        ),
        rngs=rngs,
    )

    # Train and evaluate a UBM:
    trainer = Trainer(
        optax.adamw(0.0003),
        epochs=10,
        early_stopping=EarlyStopping(patience=0),
    )
    train_df = trainer.train(model, train_loader, val_loader)
    test_df = trainer.test_clicks(model, test_loader)


if __name__ == "__main__":
    main()
