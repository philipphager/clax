from typing import Tuple

import numpy as np
from torch.utils.data import IterableDataset

from clax.datasets.utils import SessionCollator


class DummyDataset(IterableDataset):
    def __init__(
        self,
        session_range: Tuple[int, int],
        max_positions: int = 10,
        max_query_doc_id: int = 120_000_000,
        **kwargs,
    ):
        self.session_range = session_range
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

        n_sessions = self.session_range[1] - self.session_range[0]
        self.query_doc_ids = np.random.randint(
            max_query_doc_id, size=(n_sessions, max_positions), dtype=np.int32
        )
        self.clicks = np.random.randint(
            2, size=(n_sessions, max_positions), dtype=np.bool_
        )

    def __len__(self) -> int:
        return self.session_range[1] - self.session_range[0]

    def __iter__(self):
        n_sessions = self.session_range[1] - self.session_range[0]
        for i in range(n_sessions):
            yield {
                "query_doc_ids": self.query_doc_ids[i % 1000],
                "clicks": np.ones_like(self.positions, dtype=np.float16),
                "positions": self.positions,
                "mask": np.ones_like(self.positions, dtype=np.bool_),
                "n": 10,
            }
