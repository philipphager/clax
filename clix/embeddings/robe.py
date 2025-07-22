from dataclasses import dataclass

import jax.numpy as jnp
from flax import nnx
from flax.typing import Initializer

from clix.embeddings.base import EmbeddingConfig
from clix.embeddings.utils import UniversalHash


@dataclass
class RobeDEmbeddingConfig(EmbeddingConfig):
    compression_ratio: int = 100

    def create_embedding(self, num_embeddings: int, rngs: nnx.Rngs) -> nnx.Module:
        return RobeDEmbedding(
            num_embeddings=num_embeddings,
            features=self.features,
            embedding_init=self.embedding_init,
            compression_ratio=self.compression_ratio,
            rngs=rngs,
        )


class RobeDEmbedding(nnx.Module):
    """
    A basic ROBE-D embedding layer. In contrast to hashing items to shared embeddings,
    ROBE uses a single, shared embedding memory and embeddings can overlap.

    References:
    Desai, Li, and Shrivastava (2021). "Random offset block embedding array (robe) for criteotb benchmark mlperf dlrm model..."
    """

    def __init__(
        self,
        num_embeddings: int,
        features: int,
        embedding_init: Initializer,
        *,
        compression_ratio: int,
        rngs: nnx.Rngs,
    ):
        self.features = features
        self.robe_array_size = max(1, num_embeddings * features // compression_ratio)
        self.robe_array = nnx.Param(
            embedding_init(rngs.params(), (self.robe_array_size, 1)).ravel()
        )
        self.block_start_hash_fn = UniversalHash(
            max_output=self.robe_array_size,
            num_args=1,
            rngs=rngs,
        )
        self.sign_hash_fn = UniversalHash(
            max_output=2,
            num_args=2,
            rngs=rngs,
        )

    def __call__(self, idx):
        block_start_idx = self.block_start_hash_fn(idx)

        offsets = jnp.arange(self.features)
        block_idx = (block_start_idx[..., None] + offsets) % self.robe_array_size
        embeddings = self.robe_array[block_idx]

        # Scale hash in {0, 1} to {-1, 1}:
        signs = 2 * self.sign_hash_fn(idx[..., None], offsets[None, :]) - 1

        return signs * embeddings
