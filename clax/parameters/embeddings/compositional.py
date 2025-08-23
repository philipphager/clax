from enum import Enum
from typing import Union

import jax.numpy as jnp
import math
from flax import nnx
from flax.typing import Initializer

from clax.parameters.embeddings.utils import UniversalHash


class Combination(str, Enum):
    ADDITION = "addition"
    MULTIPLICATION = "multiplication"
    CONCATENATION = "concatenation"


class QREmbedding(nnx.Module):
    def __init__(
        self,
        num_embeddings: int,
        features: int,
        embedding_init: Initializer,
        compression_ratio: int = 10,
        shuffle_indices: bool = True,
        *,
        qr_combination: Union[Combination, str] = Combination.MULTIPLICATION,
        rngs: nnx.Rngs,
    ):
        self.compression_ratio = compression_ratio
        self.num_quotient_embeddings = math.ceil(num_embeddings / compression_ratio)
        self.shuffle_indices = shuffle_indices

        self.quotient_embedding = nnx.Embed(
            num_embeddings=self.num_quotient_embeddings,
            features=features,
            embedding_init=embedding_init,
            rngs=rngs,
        )
        self.remainder_embedding = nnx.Embed(
            num_embeddings=self.compression_ratio,
            features=features,
            embedding_init=embedding_init,
            rngs=rngs,
        )

        if self.shuffle_indices:
            self.shuffle_fn = UniversalHash(
                max_output=num_embeddings,
                num_args=1,
                rngs=rngs,
            )
        else:
            self.shuffle_fn = lambda x: x

        if qr_combination == Combination.MULTIPLICATION:
            self.combine_fn = lambda q, r: q * r
        elif qr_combination == Combination.ADDITION:
            self.combine_fn = lambda q, r: q + r
        elif qr_combination == Combination.CONCATENATION:
            self.projection = nnx.Linear(2 * features, features, rngs=rngs)
            self.combine_fn = lambda q, r: self.projection(
                jnp.concatenate([q, r], axis=-1)
            )
        else:
            raise ValueError(f"Unknown combination type: {qr_combination}")

    def __call__(self, idx):
        # Applying a hash function with low collision prob. to shuffle indices:
        idx = self.shuffle_fn(idx)

        quotient_idx = idx // self.compression_ratio
        remainder_idx = idx % self.compression_ratio

        quotient_embed = self.quotient_embedding(quotient_idx)
        remainder_embed = self.remainder_embedding(remainder_idx)

        return self.combine_fn(quotient_embed, remainder_embed)
