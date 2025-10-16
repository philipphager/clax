from pathlib import Path
from typing import Tuple

import optax
from flax import nnx
from flax.training.early_stopping import EarlyStopping
from torch.utils.data import DataLoader

from clax import UserBrowsingModel
from clax.datasets import YandexDataset
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
    # Load a few sessions from the Yandex WSCD-2012 dataset:
    dataset_dir = Path("../../clax-datasets/yandex")
    train_loader = get_yandex_loader(dataset_dir, session_range=(0, 1_000_000))
    val_loader = get_yandex_loader(dataset_dir, session_range=(1_000_000, 1_500_000))
    test_loader = get_yandex_loader(dataset_dir, session_range=(1_500_000, 2_000_000))

    # Instantiate a UBM:
    rngs = nnx.Rngs(42)
    model = UserBrowsingModel(
        query_doc_pairs=10_000_000,
        positions=10,
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

    # Use the trained UBM:
    batch = next(iter(test_loader))

    print("Predict unconditional click probabilities:")
    print(model.predict_clicks(batch))

    print("Predict conditional click probabilities:")
    print(model.predict_conditional_clicks(batch))

    print("Predict query-doc relevance for ranking:")
    print(model.predict_relevance(batch))

    print("Sample clicks:")
    print(model.sample(batch, rngs=rngs))


if __name__ == "__main__":
    main()
