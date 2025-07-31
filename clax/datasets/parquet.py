from pathlib import Path
from typing import List, Tuple, Union, Optional, Set

import numpy as np
import pyarrow.dataset as ds
import pyarrow.parquet as pq
import torch
from torch.utils.data import IterableDataset

from clax.datasets.utils import SessionCollator

FileRangeTuple = Tuple[Path, int, int]


class ParquetDataset(IterableDataset):
    def __init__(
        self,
        source: Union[List[Union[Path, str]], Union[Path, str]],
        session_range: Tuple[int, int],
        max_positions: int = 10,
        file_glob: str = "*.parquet",
        filter_query_ids: Optional[Set[int]] = None,
    ):
        """
        A PyTorch IterableDataset for CLAX datasets.

        This class handles large-scale, multi-file Parquet datasets where each row
        is a user session, without loading entire files into RAM.

        The dataset automatically distributes the workload among PyTorch DataLoader
        workers, ensuring each worker processes a unique subset of the files and
        session ranges.
        """
        files = self._find_files(source, file_glob)
        self.file_ranges = self._file_ranges(files, session_range)
        self.max_positions = max_positions
        self.filter_query_ids = filter_query_ids
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

        for path, begin_row, end_row in file_ranges:
            file = pq.ParquetFile(path)
            rows_processed = 0

            has_query_id = "query_id" in file.schema_arrow.names
            if self.filter_query_ids and not has_query_id:
                raise ValueError(
                    "A set of query ids was provided for filtering sessions, "
                    "but the file does not contain a 'query_id' column."
                )

            for batch in file.iter_batches():
                batch_size = len(batch)
                overlap_begin = max(0, begin_row - rows_processed)
                overlap_end = min(batch_size, end_row - rows_processed)

                if overlap_begin < overlap_end:
                    query_doc_ids_batch = batch["query_doc_ids"].to_numpy(
                        zero_copy_only=False
                    )
                    clicks_batch = batch["clicks"].to_numpy(zero_copy_only=False)

                    query_ids = None

                    if self.filter_query_ids:
                        query_ids = batch["query_id"].to_numpy(zero_copy_only=False)

                    for i in range(overlap_begin, overlap_end):
                        if (
                            self.filter_query_ids
                            and query_ids[i] not in self.filter_query_ids
                        ):
                            # Skip user session as its query_id is not in the provided set.
                            continue

                        query_doc_ids = query_doc_ids_batch[i]
                        clicks = clicks_batch[i]
                        n = min(len(query_doc_ids), self.max_positions)

                        yield {
                            "query_doc_ids": query_doc_ids[:n],
                            "clicks": clicks[:n],
                            "mask": self.mask[:n],
                            "positions": self.positions[:n],
                            "n": n,
                        }

                rows_processed += batch_size

                if rows_processed >= end_row:
                    # Reached the end of the current file range that should be parsed,
                    # Break to skip to the next file.
                    break

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

        if len(self.file_ranges) > 1:
            # Multiple files are distributed amongst workers.
            # Select the files for the current worker:
            return [
                f for i, f in enumerate(self.file_ranges) if i % workers == worker_id
            ]
        elif len(self.file_ranges) == 1:
            # A single file is split amongst workers.
            # Select the range of rows for the current worker:
            file_path, total_begin_row, total_end_row = self.file_ranges[0]
            total_rows = total_end_row - total_begin_row

            worker_rows = total_rows // workers
            remainder = total_rows % workers

            begin_row = (
                total_begin_row + worker_id * worker_rows + min(worker_id, remainder)
            )
            end_row = begin_row + worker_rows + (1 if worker_id < remainder else 0)
            return [(file_path, begin_row, end_row)]

    @staticmethod
    def _find_files(
        source: Union[List[Path], Path, str],
        file_glob: Optional[str] = None,
    ) -> List[Path]:
        """
        Finds and validates a paths to parquet files. If a directory is submitted,
        the method searches for files with glob(file_glob). The resulting list of
        paths is sorted alphabetically.
        """
        if isinstance(source, list):
            paths = [Path(p) for p in source]
        else:
            path = Path(source)
            paths = list(path.glob(file_glob)) if path.is_dir() else [path]

        for path in paths:
            if not path.exists():
                raise FileNotFoundError(f"No such file: '{path}'")

        return sorted(paths)

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
            dataset = ds.dataset(file)
            num_sessions = dataset.count_rows()

            file_begin = total_sessions
            file_end = total_sessions + num_sessions

            overlap_begin = max(file_begin, session_begin)
            overlap_end = min(file_end, session_end)

            if overlap_begin < overlap_end:
                begin_row = overlap_begin - total_sessions
                end_row = overlap_end - total_sessions
                file_ranges.append((file, begin_row, end_row))

            if total_sessions >= session_end:
                break

            total_sessions += num_sessions

        return file_ranges
