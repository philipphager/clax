from collections import defaultdict
from typing import Dict, List, Union

import hydra
import jax
import numpy as np
import optax
import pyarrow_hotfix
from datasets import load_dataset
from flax import nnx
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from clax.datasets.utils import pad
from clax.models import PositionBasedModel
from clax.parameters import (
    DeepParameterConfig,
    EmbeddingParameterConfig,
    LinearParameterConfig,
)
from clax.trainer import Trainer

pyarrow_hotfix.uninstall()


class BaiduULTRCollator:
    def __init__(
        self,
        query_features: Dict[str, np.dtype],
        doc_features: Dict[str, np.dtype],
        ltr_features: Dict[str, np.dtype],
        rename_columns: Dict[str, str],
    ):
        self.query_features = query_features
        self.doc_features = doc_features
        self.ltr_features = ltr_features
        self.rename_columns = rename_columns

    def __call__(
        self, samples: List[Dict[str, Union[np.ndarray, int]]]
    ) -> Dict[str, np.ndarray]:
        batch = {}
        ltr_features = []

        # Aggregate query features, scalar values per query/session: (batch,)
        for feature, dtype in self.query_features.items():
            batch[feature] = np.array([s[feature] for s in samples], dtype=dtype)

        max_n = batch["n"].max()

        # Aggregate scalar document features: (batch, documents)
        for feature, dtype in self.doc_features.items():
            batch[feature] = pad(samples, feature, max_n, dtype=dtype)

        batch["mask"] = np.ones((len(samples), max_n), dtype=np.bool_)

        # Stack ltr features into a vector per document: (batch, documents, features)
        for feature, dtype in self.ltr_features.items():
            ltr_features.append(pad(samples, feature, max_n, dtype=dtype))

        batch["ltr_features"] = np.stack(ltr_features, axis=-1)

        for old_name, new_name in self.rename_columns.items():
            batch[new_name] = batch[old_name]
            del batch[old_name]

        return batch


import torch
from typing import List
from collections import defaultdict
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader


def collate_fn(samples: List):
    batch = defaultdict(lambda: [])

    for sample in samples:
        batch["position"].append(sample["position"])
        batch["click"].append(sample["click"])
        batch["n"].append(sample["n"])

    return {
        "position": pad_sequence(batch["position"], batch_first=True),
        "click": pad_sequence(batch["click"], batch_first=True),
        "n": torch.tensor(batch["n"]),
    }


@hydra.main(version_base="1.3", config_path="clax/config/", config_name="config")
def main(config: DictConfig):
    train_clicks = load_dataset(
        "philipphager/baidu-ultr_baidu-mlm-ctr",
        name="clicks",
        split="train",
        cache_dir="/ivi/ilps/personal/phager/huggingface/",
        trust_remote_code=True,
    ).with_format("numpy")

    test_clicks = load_dataset(
        "philipphager/baidu-ultr_baidu-mlm-ctr",
        name="clicks",
        split="test",
        cache_dir="/ivi/ilps/personal/phager/huggingface/",
        trust_remote_code=True,
    ).with_format("numpy")

    collate_fn = BaiduULTRCollator(
        query_features={"n": np.int64},
        doc_features={"position": np.int16, "click": np.float16},
        ltr_features={
            "bm25": np.float32,
            "bm25_title": np.float32,
            "bm25_abstract": np.float32,
            "tf_idf": np.float32,
            "tf": np.float32,
            "idf": np.float32,
            "ql_jelinek_mercer_short": np.float32,
            "ql_jelinek_mercer_long": np.float32,
            "ql_dirichlet": np.float32,
            "document_length": np.int32,
            "title_length": np.int32,
            "abstract_length": np.int32,
        },
        rename_columns={"click": "clicks", "position": "positions"},
    )

    train_clicks = DataLoader(
        train_clicks,
        batch_size=512,
        collate_fn=collate_fn,
        num_workers=6,
        persistent_workers=True,
    )
    test_clicks = DataLoader(
        test_clicks,
        batch_size=512,
        collate_fn=collate_fn,
        num_workers=6,
        persistent_workers=True,
    )

    model = PositionBasedModel(
        examination=EmbeddingParameterConfig(use_feature="positions", parameters=22),
        attraction=LinearParameterConfig(use_feature="ltr_features", features=12),
        rngs=nnx.Rngs(0),
    )

    trainer = Trainer(optax.adamw(0.003), epochs=100, patience=0)
    train_df = trainer.train(model, train_clicks, test_clicks)


if __name__ == "__main__":
    main()
