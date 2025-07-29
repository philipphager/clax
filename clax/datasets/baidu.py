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
            # Initialize an empty Polars DataFrame if no files are loaded
            self.df = pl.DataFrame({"query_doc_ids": [], "clicks": []}).with_columns(
                pl.Series([], dtype=pl.List(pl.Int32)).alias("query_doc_ids"),
                pl.Series([], dtype=pl.List(pl.Float32)).alias(
                    "clicks"
                ),  # Polars Float32 maps to numpy float16 well
            )
        else:
            print(
                f"Loading {len(files_to_load)} parquet files into Polars DataFrame..."
            )
            # Load data into a single Polars DataFrame
            # Explicitly select only the necessary columns to reduce memory footprint.
            self.df = (
                pl.scan_parquet(files_to_load)
                .select(["query_doc_ids", "clicks"])
                .collect()
            )
            print(f"Loaded {len(self.df)} sessions into Polars DataFrame.")

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
        return len(self.df)

    def __getitem__(self, idx):
        # Access the row directly from the Polars DataFrame
        # .row(idx) returns a tuple, which we then convert to NumPy arrays.
        # This performs the NumPy conversion on a per-item basis.
        row_data = self.df.row(idx, named=False)
        query_doc_ids_list = row_data[0]
        clicks_list = row_data[1]

        # Convert to NumPy arrays with specified dtypes
        query_doc_ids = np.array(query_doc_ids_list, dtype=np.int32)
        clicks = np.array(clicks_list, dtype=np.float16)

        # Ensure 'n' does not exceed max_positions
        n = min(len(query_doc_ids), self.max_positions)
        return {
            "query_doc_ids": query_doc_ids[:n],
            "clicks": clicks[:n],
            "mask": self.mask[:n],
            "positions": self.positions[:n],
            "n": n,
        }
