from pathlib import Path
from typing import Tuple, Union, Dict, Any, Generator, Set

import numpy as np
from pyarrow import RecordBatch

from clax.datasets.parquet import BaseParquetDataset
from clax.datasets.utils import SessionCollator


class BaiduUltrDataset(BaseParquetDataset):
    def __init__(
        self,
        dataset_dir: Union[Path, str],
        session_range: Tuple[int, int],
        max_positions: int = 10,
        file_glob: str = "part-*.parquet",
        **kwargs,
    ):
        files = list(Path(dataset_dir).glob(file_glob))
        super().__init__(files, session_range)

        self.max_positions = max_positions
        self.mask = np.ones(self.max_positions, dtype=np.bool_)
        self.positions = np.arange(1, self.max_positions + 1, dtype=np.int16)
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

    def get_unique_query_ids(
        self,
        min_sessions: int = 0,
    ) -> Set[int]:
        raise NotImplementedError(
            "The Baidu-ULTR dataset does not support counting query ids."
        )

    def _parse_batch(
        self,
        batch: RecordBatch,
        begin_idx: int,
        end_idx: int,
    ) -> Generator[Dict[str, Any], None, None]:
        query_doc_ids = batch["query_doc_ids"].to_numpy(zero_copy_only=False)
        clicks = batch["clicks"].to_numpy(zero_copy_only=False)

        for idx in range(begin_idx, end_idx):
            n = min(len(query_doc_ids[idx]), self.max_positions)

            yield {
                "query_doc_ids": query_doc_ids[idx][:n],
                "clicks": clicks[idx][:n],
                "mask": self.mask[:n],
                "positions": self.positions[:n],
                "n": n,
            }


class BaiduUltrFeatureClickDataset(BaseParquetDataset):
    def __init__(
        self,
        dataset_dir: Union[Path, str],
        session_range: Tuple[int, int],
        max_positions: int = 10,
        file_glob: str = "part-*_split-*.parquet",
        **kwargs,
    ):
        files = list(sorted(Path(dataset_dir).glob(file_glob)))
        super().__init__(files, session_range)
        self.max_positions = max_positions
        self.mask = np.ones(30, dtype=np.bool_)
        self.collate_fn = SessionCollator(
            query_features={
                "n": np.int16,
            },
            doc_features={
                "query_doc_features": np.float32,
                "positions": np.int16,
                "mask": np.bool_,
                "clicks": np.float16,
            },
        )

    def get_unique_query_ids(
        self,
        min_sessions: int = 0,
    ) -> Set[int]:
        raise NotImplementedError(
            "The Baidu-ULTR dataset does not support counting query ids."
        )

    def _parse_batch(
        self,
        batch: RecordBatch,
        begin_idx: int,
        end_idx: int,
    ) -> Generator[Dict[str, Any], None, None]:
        query_doc_features = batch["query_doc_features"].to_numpy(zero_copy_only=False)
        positions = batch["positions"].to_numpy(zero_copy_only=False)
        clicks = batch["clicks"].to_numpy(zero_copy_only=False)

        for idx in range(begin_idx, end_idx):
            n = (positions[idx] <= self.max_positions).sum()

            yield {
                "query_doc_features": np.vstack(query_doc_features[idx])[:n],
                "clicks": clicks[idx][:n],
                "positions": positions[idx][:n],
                "mask": self.mask[:n],
                "n": n,
            }


class BaiduUltrFeatureAnnotationDataset(BaseParquetDataset):
    def __init__(
        self,
        dataset_dir: Union[Path, str],
        session_range: Tuple[int, int],
        file_glob: str = "annotations.parquet",
        **kwargs,
    ):
        files = list(sorted(Path(dataset_dir).glob(file_glob)))
        super().__init__(files, session_range)

        self.mask = np.ones(120, dtype=np.bool_)
        self.collate_fn = SessionCollator(
            query_features={
                "n": np.int16,
            },
            doc_features={
                "query_doc_features": np.float32,
                "mask": np.bool_,
                "labels": np.float16,
            },
        )

    def get_unique_query_ids(
        self,
        min_sessions: int = 0,
    ) -> Set[int]:
        raise NotImplementedError(
            "The Baidu-ULTR dataset does not support counting query ids."
        )

    def _parse_batch(
        self,
        batch: RecordBatch,
        begin_idx: int,
        end_idx: int,
    ) -> Generator[Dict[str, Any], None, None]:
        query_doc_features = batch["query_doc_features"].to_numpy(zero_copy_only=False)
        labels = batch["labels"].to_numpy(zero_copy_only=False)

        for idx in range(begin_idx, end_idx):
            n = len(labels[idx])

            yield {
                "query_doc_features": np.vstack(query_doc_features[idx])[:n],
                "labels": labels[idx],
                "mask": self.mask[:n],
                "n": n,
            }
