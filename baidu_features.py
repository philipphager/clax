from pathlib import Path
from pathlib import Path
from time import perf_counter

import hydra
import optax
from flax import nnx
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from clax.datasets import BaiduUltrUvADataset
from clax.models import UserBrowsingModel
from clax.parameters import DeepParameterConfig
from clax.trainer import Trainer


@hydra.main(version_base="1.3", config_path="clax/config/", config_name="config")
def main(config: DictConfig):
    print(OmegaConf.to_yaml(config))
    rngs = nnx.Rngs(config.random_state)

    train_dataset = BaiduUltrUvADataset(
        dataset_dir="clax-datasets/baidu-ultr-uva",
        session_range=[0, 1_000_000],
    )
    val_dataset = BaiduUltrUvADataset(
        dataset_dir="clax-datasets/baidu-ultr-uva",
        session_range=[1_000_000, 1_500_000],
    )
    test_dataset = BaiduUltrUvADataset(
        dataset_dir="clax-datasets/baidu-ultr-uva",
        session_range=[1_500_000, 2_000_000],
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
        num_workers=4,
    )

    model = UserBrowsingModel(
        positions=22,
        attraction=DeepParameterConfig(
            use_feature="query_doc_features",
            features=12,
        ),
        rngs=nnx.Rngs(0),
    )

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
