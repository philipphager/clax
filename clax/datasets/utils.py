import itertools
from functools import partial
from typing import Union, List, Dict

import numpy as np


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


def batched(iterable, n):
    it = iter(iterable)
    while batch := list(itertools.islice(it, n)):
        yield batch
