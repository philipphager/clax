from pathlib import Path
from typing import List, Tuple, Union, Dict, Any, Generator, Optional, Set

import numpy as np
from pyarrow import RecordBatch

from clax.datasets.parquet import BaseParquetDataset
from clax.datasets.utils import SessionCollator


class YandexDataset(BaseParquetDataset):
    def __init__(
        self,
        dataset_dir: Union[Path, str],
        session_range: Tuple[int, int],
        filter_query_ids: Optional[List[int]] = None,
        max_positions: int = 10,
        file_glob: str = "part-*.parquet",
        **kwargs,
    ):
        files = list(Path(dataset_dir).glob(file_glob))
        super().__init__(files, session_range)

        self.max_positions = max_positions
        self.mask = np.ones(self.max_positions, dtype=np.bool_)
        self.positions = np.arange(1, self.max_positions + 1, dtype=np.int16)
        self.filter_query_ids = filter_query_ids

        self.collate_fn = SessionCollator(
            query_features={
                "query_id": np.int32,
                "n": np.int16,
            },
            doc_features={
                "query_doc_ids": np.int32,
                "positions": np.int16,
                "mask": np.bool_,
                "clicks": np.float16,
            },
        )

    def get_unique_query_ids(
        self,
        min_sessions: int = 0,
    ) -> Set[int]:
        return self._get_unique_query_ids(
            min_sessions,
            query_column="query_id",
        )

    def _parse_batch(
        self,
        batch: RecordBatch,
        begin_idx: int,
        end_idx: int,
    ) -> Generator[Dict[str, Any], None, None]:
        query_ids = batch["query_id"].to_numpy(zero_copy_only=False)
        query_doc_ids = batch["query_doc_ids"].to_numpy(zero_copy_only=False)
        clicks = batch["clicks"].to_numpy(zero_copy_only=False)

        for idx in range(begin_idx, end_idx):
            if self.filter_query_ids is None or query_ids[idx] in self.filter_query_ids:
                n = min(len(query_doc_ids[idx]), self.max_positions)

                yield {
                    "query_id": query_ids[idx],
                    "query_doc_ids": query_doc_ids[idx][:n],
                    "clicks": clicks[idx][:n],
                    "mask": self.mask[:n],
                    "positions": self.positions[:n],
                    "n": n,
                }
