import json
import math
from pathlib import Path
from typing import Optional, Dict, Tuple, Any

import numpy as np
import torch
from torch.utils.data import IterableDataset
from tqdm import tqdm

from clix.datasets import SessionCollator


def build_index(path: Path, per_partition: int, query_indicator: bytes = b"\tQ\t"):
    print(
        f"Creating index for {path}, storing access every {per_partition:_} sessions..."
    )
    partition_begins = []
    partition_ends = []
    total_sessions = 0
    bytes_since_last_index = 0

    with open(path, "rb") as f:
        file_size = path.stat().st_size
        progress_bar = tqdm(total=file_size, unit="B", unit_scale=True)

        while True:
            byte_position = f.tell()
            line = f.readline()

            if not line:
                break

            bytes_since_last_index += len(line)

            if line.find(query_indicator) != -1:
                if total_sessions % per_partition == 0:
                    if len(partition_begins) > len(partition_ends):
                        partition_ends.append(byte_position)

                    partition_begins.append(byte_position)

                    progress_bar.update(bytes_since_last_index)
                    bytes_since_last_index = 0

                total_sessions += 1

        partition_ends.append(byte_position)
        progress_bar.close()

    return {
        "sessions_per_partition": per_partition,
        "total_sessions": total_sessions,
        "partition_begins": partition_begins,
        "partition_ends": partition_ends,
        "total_partitions": len(partition_begins),
    }


class YandexDataset(IterableDataset):

    def __init__(
        self,
        path: Path,
        index_path: Path,
        session_range: Optional[Tuple[int, int]] = None,
        max_positions: int = 10,
        buffer_size_mb: int = 8,
    ):
        assert path is not None and path.exists()
        assert index_path is not None and index_path.exists()

        self.path = path
        self.index = self._load_index(index_path)
        self.session_range = session_range
        self.partitions = self._get_partitions(self.index, session_range)
        self.max_positions = max_positions
        self.buffer_size = buffer_size_mb * 1024 * 1024

        # Pre-compute reusable outputs:
        self.mask = np.ones(self.max_positions, dtype=np.bool_)
        self.positions = np.arange(1, self.max_positions + 1, dtype=np.int16)

        self.collate_fn = SessionCollator(
            query_features={"query_id": np.int32, "n": np.int16},
            doc_features={
                "query_doc_ids": np.int32,
                "positions": np.int16,
                "mask": np.bool_,
                "clicks": np.float16,
            },
        )

    def __iter__(self):
        local_partitions = self._get_local_partitions()

        with open(self.path, "rb", buffering=self.buffer_size) as file:
            for partition in local_partitions:
                yield from self._parse_partition(file, partition)

    def __len__(self):
        return self.session_range[1] - self.session_range[0]

    def _parse_partition(self, file, partition):
        partition_begin = self.index["partition_begins"][partition]
        partition_end = self.index["partition_ends"][partition]

        # Move to partition start
        file.seek(partition_begin)
        current_query = None
        current_session = None
        bytes_read = 0

        while bytes_read < (partition_end - partition_begin):
            line = file.readline()

            if not line:
                # Reached end of the file
                break

            bytes_read += len(line)

            if partition_begin + bytes_read > partition_end:
                # End if we've exceeded partition boundary
                break

            columns = line.rstrip(b"\n").split(b"\t")

            if columns[2] == b"Q":
                # Yield previous session if it exists
                if current_query is not None and current_session is not None:
                    yield current_session

                # Start new session
                current_query = int(columns[3])

                # Parse query-document-ids (columns 5 onwards)
                query_doc_ids = np.array(columns[5:], dtype=np.int32)
                n = len(query_doc_ids)

                current_session = {
                    "query_id": current_query,
                    "query_doc_ids": query_doc_ids,
                    "clicks": np.zeros(n, dtype=np.float16),
                    "n": n,
                    "mask": self.mask[:n],
                    "positions": self.positions[:n],
                }
                doc2index = {doc_id: i for i, doc_id in enumerate(query_doc_ids)}
            elif columns[2] == b"C":
                # Parse click event
                clicked_doc_id = int(columns[3])
                idx = doc2index.get(clicked_doc_id)

                if idx is not None:
                    current_session["clicks"][idx] = 1.0

        # Yield final query in partition
        if current_query is not None and current_session is not None:
            yield current_session

    def _get_local_partitions(self):
        worker_info = torch.utils.data.get_worker_info()

        if worker_info is None:
            total_workers = 1
            worker_id = 0
        else:
            total_workers = worker_info.num_workers
            worker_id = worker_info.id

        # Distribute partitions across workers
        return [p for p in self.partitions if p % total_workers == worker_id]

    @staticmethod
    def _load_index(index_path: Path) -> Dict:
        with open(index_path) as f:
            return json.load(f)

    @staticmethod
    def _get_partitions(
        index: Dict[str, Any],
        session_range: Optional[Tuple[int, int]],
    ):
        """
        Get partitions based on session range: (begin_sessions, end_sessions).
        """
        if session_range is None:
            return list(range(index["total_partitions"]))

        session_begin, session_end = session_range
        session_end = min(session_end, index["total_sessions"])
        sessions_per_partition = index["sessions_per_partition"]

        partition_begin = session_begin // sessions_per_partition
        partition_end = math.ceil(session_end / sessions_per_partition)
        partition_end = min(partition_end, index["total_partitions"])

        return list(range(partition_begin, partition_end))
