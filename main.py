from pathlib import Path
from time import perf_counter

import hydra
import optax
from flax import nnx
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

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
        persistent_workers=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.eval_batch_size,
        collate_fn=val_dataset.collate_fn,
        num_workers=8,
        persistent_workers=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.eval_batch_size,
        collate_fn=test_dataset.collate_fn,
        num_workers=4,
    )

    model_fn = instantiate(config.model)
    model = model_fn(rngs=rngs)

    trainer = Trainer(optax.adamw(0.003), epochs=100, patience=0)

    timer_start = perf_counter()
    train_df = trainer.train(model, train_loader, val_loader)
    timer_stop = perf_counter()
    test_df = trainer.test(model, test_loader)
    test_df["train_time_s"] = timer_stop - timer_start

    result_dir = Path(f"results/{config.experiment}")
    result_dir.mkdir(exist_ok=True, parents=True)

    train_df.to_csv(result_dir / f"train_{model.name.lower()}.csv", index=False)
    test_df.to_csv(result_dir / f"test_{model.name.lower()}.csv", index=False)


if __name__ == "__main__":
    main()
