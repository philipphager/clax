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
    """
    Pads a list of features to the same length (max_n).

    Handles 1D features (e.g., clicks) resulting in a 2D padded array
    and 2D document features (e.g., embeddings) resulting in a 3D padded array.
    """
    batch_size = len(samples)

    # 1. Inspect the first sample's feature to get the shape of individual items.
    # For a 1D (docs,) array, this will be ().
    # For a 2D (docs, features) array, this will be (features,).
    first_item = samples[0][feature]
    feature_shape = first_item.shape[1:]

    padded_shape = (batch_size, max_n) + feature_shape
    array = np.zeros(padded_shape, dtype=dtype)

    for row, sample in enumerate(samples):
        n = min(len(sample[feature]), max_n)
        array[row, :n] = sample[feature][:max_n]

    return array
