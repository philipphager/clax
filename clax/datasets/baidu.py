from pathlib import Path
from typing import List, Tuple, Union

import numpy as np
import polars as pl
import pyarrow.parquet as pq
import torch
from torch.utils.data import IterableDataset

from clax.datasets.utils import SessionCollator, batched

FileRangeTuple = Tuple[Path, int, int]


class BaiduULTRDataset(IterableDataset):
    def __init__(
        self,
        path: Union[Path, str],
        session_range: Tuple[int, int],
        max_positions: int = 10,
        file_batch_size: int = 5,
    ):
        path = Path(path)
        files = self._find_files(path)
        self.file_ranges = self._file_ranges(files, session_range)
        self.max_positions = max_positions
        self.file_batch_size = file_batch_size
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
        total_sessions = 0

        for _, begin_row, end_row in self.file_ranges:
            total_sessions += end_row - begin_row

        return total_sessions

    def __iter__(self):
        file_ranges = self._get_local_file_ranges()
        batched_file_ranges = batched(file_ranges, self.file_batch_size)

        for file_ranges in batched_file_ranges:
            dfs = []

            for file, begin_row, end_row in file_ranges:
                n_rows = end_row - begin_row
                df = pl.scan_parquet(file).slice(begin_row, n_rows)
                dfs.append(df)

            df = pl.concat(dfs).collect()

            for query_doc_ids, clicks in zip(
                df["query_doc_ids"].to_numpy(),
                df["clicks"].to_numpy(),
            ):
                n = min(len(query_doc_ids), self.max_positions)

                yield {
                    "query_doc_ids": query_doc_ids[:n],
                    "clicks": clicks[:n],
                    "mask": self.mask[:n],
                    "positions": self.positions[:n],
                    "n": n,
                }

    def _get_local_file_ranges(self) -> List[FileRangeTuple]:
        """
        Select a subset of file ranges to iterate, based on the current worker process.
        See: https://pytorch.org/docs/stable/data.html#torch.utils.data.IterableDataset
        """
        info = torch.utils.data.get_worker_info()

        if info is None:
            workers = 1
            worker_id = 0
        else:
            workers = info.num_workers
            worker_id = info.id

        return [f for i, f in enumerate(self.file_ranges) if i % workers == worker_id]

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
        return list(path.glob("part-*.parquet"))
