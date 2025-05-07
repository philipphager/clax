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

    def __getitem__(self, idx):
        query_id = self.query_ids[idx]
        start, end = self.query_ranges[idx]

        return {
            "query_id": query_id,
            "query_doc_ids": self.query_doc_ids[start:end],
            "positions": self.positions[start:end],
            "clicks": self.clicks[start:end],
            "n": end - start,
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

    @staticmethod
    def collate_fn(samples: List[Dict[str, Union[np.ndarray, int]]]):
        query_ids = np.array([sample["query_id"] for sample in samples], dtype=np.int32)
        n_values = np.array([sample["n"] for sample in samples], dtype=np.int16)

        batch = {
            "query_id": query_ids,
            "mask": create_mask(n_values),
            "n": n_values,
        }

        # Add padded columns with appropriate dtypes:
        padding_config = {
            "query_doc_ids": np.int32,
            "positions": np.int16,
            "clicks": np.float16,
        }

        for col, dtype in padding_config.items():
            batch[col] = pad(samples, col, n_values.max(), dtype)

        return batch


def create_mask(n_values: np.ndarray):
    batch_size = len(n_values)
    mask = np.zeros((batch_size, n_values.max()), dtype=np.bool_)

    for i, n in enumerate(n_values):
        mask[i, :n] = True

    return mask


def pad(samples: List[Dict[str, np.ndarray]], column: str, max_n: int, dtype: np.dtype):
    batch_size = len(samples)
    array = np.zeros((batch_size, max_n), dtype=dtype)

    for i, sample in enumerate(samples):
        array[i, : sample["n"]] = sample[column]

    return array
