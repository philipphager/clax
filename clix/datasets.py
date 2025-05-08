from dataclasses import dataclass
from pathlib import Path
from typing import Union, List, Dict

import numpy as np
import pandas as pd
from torch.utils.data import Dataset


class YandexDataset(Dataset):
    def __init__(
        self,
        path: Union[str, Path],
    ):
        self.path = path
        self._parse_file(path)
        self.collate_fn = SessionCollator(
            query_features={"query_id": np.int32, "n": np.int16},
            doc_features={
                "query_doc_ids": np.int32,
                "positions": np.int16,
                "mask": np.bool_,
                "clicks": np.float16,
            },
        )

    def __getitem__(self, idx):
        query_id = self.query_ids[idx]
        start, end = self.query_ranges[idx]
        n = end - start

        return {
            "query_id": query_id,
            "query_doc_ids": self.query_doc_ids[start:end],
            "positions": self.positions[start:end],
            "clicks": self.clicks[start:end],
            "mask": self.mask[:n],
            "n": n,
        }

    def __len__(self):
        return len(self.query_ids)

    def _parse_file(self, path: Union[str, Path]):
        df = pd.read_csv(path)

        # Determine ranges of documents/rows belonging to the same user session,
        # either by changing query_id or by a decreasing position: 1, 2, 3 | 1, 2, ...
        query_ids = df["ROUND_ID"].to_numpy()
        positions = df["RANK"].to_numpy()
        query_changes = np.flatnonzero(np.diff(query_ids)) + 1
        position_resets = np.flatnonzero(np.diff(positions) < 0) + 1

        session_changes = np.concatenate((query_changes, position_resets))
        session_changes = np.unique(session_changes)
        session_starts = np.concatenate(([0], session_changes))
        session_ends = np.concatenate((session_changes, [len(query_ids)]))

        self.query_ids = query_ids[session_starts]
        self.query_ranges = list(zip(session_starts, session_ends))
        self.query_doc_ids = df["ITEM_ID"].to_numpy()
        self.positions = df["RANK"].to_numpy()
        self.clicks = df["CLICK"].to_numpy()
        self.mask = np.ones((self.positions.max(),), dtype=np.bool_)


class SessionCollator:
    def __init__(
        self,
        query_features: Dict[str, np.dtype],
        doc_features: Dict[str, np.dtype],
    ):
        self.query_features = query_features
        self.doc_features = doc_features

    def __call__(
        self, samples: List[Dict[str, Union[np.ndarray, int]]]
    ) -> Dict[str, np.ndarray]:
        batch = {}

        for feature, dtype in self.query_features.items():
            batch[feature] = np.array([s[feature] for s in samples], dtype=dtype)

        max_n = batch["n"].max()

        for feature, dtype in self.doc_features.items():
            batch[feature] = pad(samples, feature, max_n, dtype=dtype)

        return batch


def pad(samples: List[Dict[str, np.ndarray]], feature: str, max_n, dtype: np.dtype):
    batch_size = len(samples)

    # Allocate empty 2D array with correct datatype:
    array = np.zeros((batch_size, max_n), dtype=dtype)

    # Fill the array with the feature values:
    for row, sample in enumerate(samples):
        array[row, : sample["n"]] = sample[feature]

    return array
