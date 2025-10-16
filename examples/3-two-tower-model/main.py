from pathlib import Path
from typing import Tuple

import optax
from flax import nnx
from flax.training.early_stopping import EarlyStopping
from torch.utils.data import DataLoader

from clax import PositionBasedModel
from clax.datasets import (
    BaiduUltrFeatureClickDataset,
    BaiduUltrFeatureAnnotationDataset,
)
from clax.parameters import DeepCrossParameterConfig
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


def main():
    # Load sessions from a subset of the Baidu-ULTR dataset with pre-processed query-doc-features:
    dataset_dir = Path("../../clax-datasets/baidu-ultr-uva")
    query_doc_features = 768

    train_loader = get_baidu_click_loader(
        dataset_dir,
        session_range=(0, 1_000_000),
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

    # Instantiate a PBM with a deep cross v2 network for document attraction,
    # note might be slow on CPU:
    rngs = nnx.Rngs(42)

    model = PositionBasedModel(
        attraction=DeepCrossParameterConfig(
            use_feature="query_doc_features",
            features=query_doc_features,
        ),
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
