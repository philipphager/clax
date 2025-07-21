import math
from enum import StrEnum

import jax.numpy as jnp
from flax import nnx
from flax.typing import Initializer


class Combination(StrEnum):
    MULTIPLICATION = "multiplication"
    ADDITION = "addition"
    CONCATENATION = "concatenation"


class QuotientRemainderEmbedding(nnx.Module):
    def __init__(
        self,
        num_embeddings: int,
        features: int,
        embedding_init: Initializer,
        *,
        num_collisions: int = 4,
        combination: Combination = Combination.CONCATENATION,
        rngs: nnx.Rngs,
    ):
        self.num_collisions = num_collisions
        self.num_quotient_embeddings = math.ceil(num_embeddings / num_collisions)
        self.quotient_embedding = nnx.Embed(
            num_embeddings=self.num_quotient_embeddings,
            features=features,
            embedding_init=embedding_init,
            rngs=rngs,
        )
        self.remainder_embedding = nnx.Embed(
            num_embeddings=self.num_collisions,
            features=features,
            embedding_init=embedding_init,
            rngs=rngs,
        )

        if combination == Combination.MULTIPLICATION:
            self.combine_fn = lambda q, r: q * r
        elif combination == Combination.ADDITION:
            self.combine_fn = lambda q, r: q + r
        elif combination == Combination.CONCATENATION:
            self.projection = nnx.Linear(2 * features, features, rngs=rngs)
            self.combine_fn = lambda q, r: self.projection(
                jnp.concatenate([q, r], axis=-1)
            )
        else:
            raise ValueError(f"Unknown combination type: {combination}")

    def __call__(self, idx):
        quotient_idx = idx // self.num_collisions
        remainder_idx = idx % self.num_collisions

        quotient_embed = self.quotient_embedding(quotient_idx)
        remainder_embed = self.remainder_embedding(remainder_idx)

        return self.combine_fn(quotient_embed, remainder_embed)
