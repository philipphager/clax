import math
from dataclasses import dataclass
from enum import StrEnum

import jax.numpy as jnp
from flax import nnx
from flax.typing import Initializer

from clix.embeddings.base import EmbeddingConfig


class EmbeddingCombination(StrEnum):
    MULTIPLICATION = "multiplication"
    ADDITION = "addition"
    CONCATENATION = "concatenation"


@dataclass
class QREmbeddingConfig(EmbeddingConfig):
    compression_ratio: int = 1_000
    combination: EmbeddingCombination = EmbeddingCombination.MULTIPLICATION

    def create_embedding(self, num_embeddings: int, rngs: nnx.Rngs) -> nnx.Module:
        return QREmbedding(
            num_embeddings=num_embeddings,
            features=self.features,
            embedding_init=self.embedding_init,
            compression_ratio=self.compression_ratio,
            combination=self.combination,
            rngs=rngs,
        )


class QREmbedding(nnx.Module):
    def __init__(
        self,
        num_embeddings: int,
        features: int,
        embedding_init: Initializer,
        *,
        compression_ratio: int,
        combination: EmbeddingCombination,
        rngs: nnx.Rngs,
    ):
        self.compression_ratio = compression_ratio
        self.num_quotient_embeddings = math.ceil(num_embeddings / compression_ratio)
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

        if combination == EmbeddingCombination.MULTIPLICATION:
            self.combine_fn = lambda q, r: q * r
        elif combination == EmbeddingCombination.ADDITION:
            self.combine_fn = lambda q, r: q + r
        elif combination == EmbeddingCombination.CONCATENATION:
            self.projection = nnx.Linear(2 * features, features, rngs=rngs)
            self.combine_fn = lambda q, r: self.projection(
                jnp.concatenate([q, r], axis=-1)
            )
        else:
            raise ValueError(f"Unknown combination type: {combination}")

    def __call__(self, idx):
        quotient_idx = idx // self.compression_ratio
        remainder_idx = idx % self.compression_ratio

        quotient_embed = self.quotient_embedding(quotient_idx)
        remainder_embed = self.remainder_embedding(remainder_idx)

        return self.combine_fn(quotient_embed, remainder_embed)
