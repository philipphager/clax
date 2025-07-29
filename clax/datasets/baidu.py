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
        self.max_positions = max_positions

        data_path = Path(path)
        # Get all parquet files and select the range. Sorting ensures consistent order.
        all_files = sorted(list(data_path.glob("part-*.parquet")))
        files_to_load = all_files[session_range[0] : session_range[1]]

        if not files_to_load:
            print(f"No parquet files found in the specified range: {session_range}")
            # Initialize empty arrays to prevent errors if no files are loaded
            self.query_doc_ids = np.array([], dtype=np.int32)
            self.clicks = np.array([], dtype=np.float16)
        else:
            print(f"Loading {len(files_to_load)} parquet files...")
            # Use scan_parquet for optimized reading of multiple files.
            # Explicitly select only the necessary columns to reduce memory footprint.
            df = (
                pl.scan_parquet(files_to_load)
                .select(["query_doc_ids", "clicks"])
                .collect()
            )

            self.query_doc_ids = df["query_doc_ids"].to_numpy()
            self.clicks = df["clicks"].to_numpy()
            print(f"Loaded {len(self.query_doc_ids)} sessions into memory.")

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
        print("Data initialization complete!")

    def __len__(self) -> int:
        return len(self.query_doc_ids)

    def __getitem__(self, idx):
        # Ensure 'n' does not exceed max_positions
        n = min(len(self.query_doc_ids[idx]), self.max_positions)
        return {
            "query_doc_ids": self.query_doc_ids[idx][:n],
            "clicks": self.clicks[idx][:n],
            "mask": self.mask[:n],
            "positions": self.positions[:n],
            "n": n,
        }
