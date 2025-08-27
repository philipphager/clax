from abc import ABC, abstractmethod
from collections import Counter
from pathlib import Path
from typing import List, Tuple, Dict, Any, Generator, Set

import pyarrow.parquet as pq
import torch
from pyarrow import RecordBatch
from torch.utils.data import IterableDataset

FileRangeTuple = Tuple[Path, int, int]


class BaseParquetDataset(IterableDataset, ABC):
    def __init__(
        self,
        files: List[Path],
        session_range: Tuple[int, int],
    ):
        """
        A PyTorch IterableDataset for CLAX datasets.

        This class handles large-scale, multi-file Parquet datasets where each row
        is a user session, without loading entire files into RAM.

        The dataset automatically distributes the workload among PyTorch DataLoader
        workers, ensuring each worker processes a unique subset of the files and
        session ranges.
        """
        assert len(files) > 0, "No parquet files found to load."
        self.file_ranges = self._file_ranges(files, session_range)

    @abstractmethod
    def _parse_batch(
        self,
        batch: RecordBatch,
        begin_idx: int,
        end_idx: int,
    ) -> Generator[Dict[str, Any], None, None]:
        pass

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

            for batch in file.iter_batches(batch_size=500):
                batch_size = len(batch)
                overlap_begin = max(0, begin_row - rows_processed)
                overlap_end = min(batch_size, end_row - rows_processed)

                if overlap_begin < overlap_end:
                    yield from self._parse_batch(batch, overlap_begin, overlap_end)

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
        else:
            return []

    def _file_ranges(
        self,
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
            file_metadata = pq.read_metadata(file)
            num_sessions = file_metadata.num_rows

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

    def _get_unique_query_ids(
        self,
        min_sessions: int,
        query_column: str,
    ) -> Set[int]:
        query_counter = Counter()

        for path, begin_row, end_row in self.file_ranges:
            rows_processed = 0
            file = pq.ParquetFile(path)

            assert (
                query_column in file.schema_arrow.names
            ), f"Query id '{query_column}' not found in parquet file."

            for batch in file.iter_batches():
                batch_size = len(batch)
                overlap_begin = max(0, begin_row - rows_processed)
                overlap_end = min(batch_size, end_row - rows_processed)

                if overlap_begin < overlap_end:
                    query_ids = batch["query_id"].to_numpy(zero_copy_only=False)
                    query_counter.update(query_ids[overlap_begin:overlap_end])

                rows_processed += batch_size

                if rows_processed >= end_row:
                    # Reached the end of the current file range that should be parsed,
                    # Break to skip to the next file.
                    break

        return {
            query_id
            for query_id, count in query_counter.items()
            if count >= min_sessions
        }
