from pathlib import Path
from time import perf_counter

import hydra
import optax
from flax import nnx
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from clix.datasets.yandex import YandexDataset
from clix.models import PositionBasedModel, UserBrowsingModel
from clix.parameters import EmbeddingParameterConfig, HashEmbedding
from clix.trainer import Trainer

OmegaConf.register_new_resolver("eval", eval)


@hydra.main(version_base="1.3", config_path="clix/config/", config_name="config")
def main(config: DictConfig):
    print(OmegaConf.to_yaml(config))
    rngs = nnx.Rngs(config.random_state)

    path = Path("data/wscd-2012/YandexClicks.txt")
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
        num_workers=8,
        collate_fn=train_dataset.collate_fn,
        persistent_workers=True,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=512,
        num_workers=8,
        collate_fn=val_dataset.collate_fn,
        persistent_workers=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=512,
        num_workers=4,
        collate_fn=test_dataset.collate_fn,
    )

    model_fn = instantiate(config.model)
    model = model_fn(rngs=rngs)

    trainer = Trainer(optax.adamw(1e-3), epochs=50, patience=0)

    timer_start = perf_counter()
    train_df = trainer.train(model, train_loader, val_loader)
    timer_stop = perf_counter()
    train_df["train_time_s"] = timer_stop - timer_start
    test_df = trainer.test(model, test_loader)

    train_df.to_csv("train.csv")
    test_df.to_csv("test.csv")


if __name__ == "__main__":
    main()
