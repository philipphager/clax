from pathlib import Path
from typing import List, Tuple, Union

import numpy as np
import polars as pl
import pyarrow.dataset as ds
import pyarrow.parquet as pq
import torch
from torch.utils.data import IterableDataset

from .utils import SessionCollator, batched

FileRangeTuple = Tuple[Path, int, int]


class BaiduULTRDataset(IterableDataset):
    def __init__(
        self,
        path: Union[Path, str],
        session_range: Tuple[int, int],
        max_positions: int = 10,
    ):
        self.session_range = session_range
        self.path = Path(path)
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
        self.max_positions = max_positions
        self.mask = np.ones(self.max_positions, dtype=np.bool_)
        self.positions = np.arange(1, self.max_positions + 1, dtype=np.int16)

    def __len__(self) -> int:
        return self.session_range[1] - self.session_range[0]

    def __iter__(self):
        for file in Path(self.path).glob("*.parquet"):
            file = pq.ParquetFile(file)

            for batch in file.iter_batches():
                query_doc_ids = batch["query_doc_ids"].to_numpy(zero_copy_only=False)
                clicks = batch["clicks"].to_numpy(zero_copy_only=False)

                for i in range(len(query_doc_ids)):
                    n = min(self.max_positions, len(query_doc_ids[i]))

                    yield {
                        "query_doc_ids": query_doc_ids[i][:n],
                        "clicks": clicks[i][:n],
                        "mask": self.mask[:n],
                        "positions": self.positions[:n],
                        "n": n,
                    }
