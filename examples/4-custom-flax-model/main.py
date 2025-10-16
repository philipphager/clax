from pathlib import Path
from typing import Tuple, Dict

import optax
from flax import nnx
from flax.training.early_stopping import EarlyStopping
from jax import Array
from torch.utils.data import DataLoader

from clax import PositionBasedModel
from clax.datasets import (
    BaiduUltrFeatureClickDataset,
    BaiduUltrFeatureAnnotationDataset,
)
from clax.trainer import Trainer


def get_baidu_click_loader(
    dataset_dir: Path,
    session_range: Tuple[int, int],
):
    dataset = BaiduUltrFeatureClickDataset(
        dataset_dir=dataset_dir,
        session_range=session_range,
    )

    return DataLoader(
        dataset,
        batch_size=256,
        collate_fn=dataset.collate_fn,
        num_workers=2,
        persistent_workers=True,
    )


def get_baidu_annotation_loader(
    dataset_dir: Path,
    session_range: Tuple[int, int],
):
    dataset = BaiduUltrFeatureAnnotationDataset(
        dataset_dir=dataset_dir,
        session_range=session_range,
    )

    return DataLoader(
        dataset,
        batch_size=256,
        collate_fn=dataset.collate_fn,
        num_workers=2,
        persistent_workers=True,
    )


class CustomAttraction(nnx.Module):
    """
    Example of a custom flax module with attention,
    every module needs to specify how to compute a logit,
    log probability and probability for a given batch.

    In the simplest case, the logit layer can be re-used for probability
    and log probability computation.
    """

    def __init__(self, query_doc_features, rngs):
        super().__init__()
        self.attention = nnx.MultiHeadAttention(
            num_heads=1,
            in_features=query_doc_features,
            qkv_features=8,
            decode=False,
            rngs=rngs,
        )
        self.projection = nnx.Linear(query_doc_features, 1, rngs=rngs)

    def logit(self, batch: Dict) -> Array:
        return self.projection(self.attention(batch["query_doc_features"])).squeeze()

    def prob(self, batch: Dict) -> Array:
        return nnx.sigmoid(self.logit(batch))

    def log_prob(self, batch: Dict) -> Array:
        return nnx.log_sigmoid(self.logit(batch))


def main():
    # Load sessions from a subset of the Baidu-ULTR dataset with pre-processed query-doc-features:
    dataset_dir = Path("../../clax-datasets/baidu-ultr-uva")
    query_doc_features = 768

    train_loader = get_baidu_click_loader(
        dataset_dir,
        session_range=(0, 100_000),
    )
    val_loader = get_baidu_click_loader(
        dataset_dir,
        session_range=(1_000_000, 1_500_000),
    )
    test_loader = get_baidu_click_loader(
        dataset_dir,
        session_range=(1_500_000, 2_000_000),
    )
    annotation_loader = get_baidu_annotation_loader(
        dataset_dir,
        session_range=(0, 400_000),
    )

    # Instantiate a PBM with a custom module for document attraction,
    # note might be slow on CPU:
    rngs = nnx.Rngs(42)

    model = PositionBasedModel(
        attraction=CustomAttraction(query_doc_features, rngs),
        positions=10,
        rngs=rngs,
    )
    trainer = Trainer(
        optax.adamw(0.0003),
        epochs=3,
        early_stopping=EarlyStopping(patience=0),
    )
    train_df = trainer.train(model, train_loader, val_loader)
    click_df = trainer.test_clicks(model, test_loader)
    ranking_df = trainer.test_ranking(model, annotation_loader)


if __name__ == "__main__":
    main()
