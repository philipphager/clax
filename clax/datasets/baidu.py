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
        files = self._find_files(path)
        file_ranges = self._file_ranges(files, session_range)
        df = self.load_data(file_ranges)

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

    def load_data(self, file_ranges):
        """
        file_specs: list of tuples [(filepath, start_row, end_row), ...]
        """
        lazy_frames = []

        for file, start_row, end_row in file_ranges:
            n_rows = end_row - start_row
            lazy_df = pl.scan_parquet(file).slice(start_row, n_rows)
            lazy_frames.append(lazy_df)

        return pl.concat(lazy_frames).collect()

    @staticmethod
    def _file_ranges(
        files: List[Path],
        session_range: Tuple[int, int],
    ) -> List[FileRangeTuple]:
        """
        Determine which files should be read (and which range in each file)
        for a given range of sessions.
        """
        file_ranges: List[FileRangeTuple] = []
        session_begin, session_end = session_range
        total_sessions = 0

        for file in sorted(files):
            df = pl.scan_parquet(file)
            num_sessions = df.select(pl.len()).collect().item()

            file_begin = total_sessions
            file_end = total_sessions + num_sessions

            overlap_begin = max(file_begin, session_begin)
            overlap_end = min(file_end, session_end)

            if overlap_begin < overlap_end:
                start_row = overlap_begin - total_sessions
                end_row = overlap_end - total_sessions
                file_ranges.append((file, start_row, end_row))

            if total_sessions >= session_end:
                break

            total_sessions += num_sessions

        return file_ranges

    @staticmethod
    def _find_files(path: Path) -> List[Path]:
        return path.glob("part-*.parquet")
