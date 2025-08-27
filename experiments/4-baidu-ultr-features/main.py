from pathlib import Path
from time import perf_counter

import hydra
import optax
from flax import nnx
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from clax.trainer import Trainer


@hydra.main(version_base="1.3", config_path=".", config_name="config")
def main(config: DictConfig):
    print(OmegaConf.to_yaml(config))
    rngs = nnx.Rngs(config.random_state)

    result_dir = Path(f"{config.result_dir}/{config.experiment}")
    result_dir.mkdir(exist_ok=True, parents=True)
    print("Saving results to: ", result_dir)

    train_dataset = instantiate(
        config.dataset.clicks,
        session_range=config.train_sessions,
    )
    val_dataset = instantiate(
        config.dataset.clicks,
        session_range=config.val_sessions,
    )
    test_dataset = instantiate(
        config.dataset.clicks,
        session_range=config.test_sessions,
    )
    relevance_dataset = instantiate(
        config.dataset.annotations,
        session_range=config.test_rel_sessions,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.train_batch_size,
        collate_fn=train_dataset.collate_fn,
        num_workers=config.num_workers,
        persistent_workers=True,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.eval_batch_size,
        collate_fn=val_dataset.collate_fn,
        num_workers=config.num_workers,
        persistent_workers=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.eval_batch_size,
        collate_fn=test_dataset.collate_fn,
        num_workers=config.num_workers,
    )
    test_relevance_loader = DataLoader(
        relevance_dataset,
        batch_size=config.eval_batch_size,
        collate_fn=relevance_dataset.collate_fn,
        num_workers=1,
    )

    model_fn = instantiate(config.model)
    model = model_fn(rngs=rngs)

    trainer = Trainer(
        optax.adamw(config.learning_rate),
        epochs=config.epochs,
        patience=config.early_stopping_patience,
    )

    timer_start = perf_counter()
    train_df = trainer.train(model, train_loader, val_loader)
    timer_stop = perf_counter()
    train_df.to_csv(result_dir / f"train_{model.name.lower()}.csv", index=False)

    test_df = trainer.test(model, test_loader)
    test_df["train_time_s"] = timer_stop - timer_start
    test_df.to_csv(result_dir / f"test_{model.name.lower()}.csv", index=False)

    test_rel_df = trainer.test_relevance(model, test_relevance_loader)
    test_rel_df.to_csv(result_dir / f"test_rel_{model.name.lower()}.csv", index=False)


if __name__ == "__main__":
    main()
