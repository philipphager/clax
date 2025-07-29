from pathlib import Path
from typing import List, Tuple, Union

import numpy as np
import polars as pl
from torch.utils.data import Dataset

from clax.datasets.utils import SessionCollator

FileRangeTuple = Tuple[Path, int, int]


class BaiduULTRDataset(Dataset):
    def __init__(
        self,
        path: Union[Path, str],
        session_range: Tuple[int, int],
        max_positions: int = 10,
    ):
        self.session_range = session_range

        path = Path(path)
        files = list(path.glob("part-*.parquet"))
        df = pl.read_parquet(files)

        self.query_doc_ids = df["query_doc_ids"].to_numpy()
        self.clicks = df["clicks"].to_numpy()
        self.max_positions = max_positions
        self.collate_fn = SessionCollator(
            query_features={
                "n": np.int16,
            },
            doc_features={
                "query_doc_ids": np.int32,
                "positions": np.int16,
                "mask": np.bool_,
                "clicks": np.float16,
            },
        )
        # Pre-compute reusable outputs:
        self.mask = np.ones(self.max_positions, dtype=np.bool_)
        self.positions = np.arange(1, self.max_positions + 1, dtype=np.int16)
        print("Data loaded!")

    def __len__(self) -> int:
        return len(self.query_doc_ids)

    def __getitem__(self, idx):
        n = min(len(self.query_doc_ids[idx]), self.max_positions)
        return {
            "query_doc_ids": self.query_doc_ids[idx][:n],
            "clicks": self.clicks[idx][:n],
            "mask": self.mask[:n],
            "positions": self.positions[:n],
            "n": n,
        }
