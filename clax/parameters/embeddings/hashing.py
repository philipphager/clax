import numpy as np
import jax.numpy as jnp
import math
from flax import nnx
from flax.typing import Initializer

from clax.parameters.embeddings.utils import UniversalHash


class HashEmbedding(nnx.Module):
    """
    Hashing trick with sign correction per embedding parameter.

    References:
    Weinberger, Dasgupta, Langford, Smola, and Attenberg (2009). "Feature hashing for large scale multitask learning"
    """

    def __init__(
        self,
        num_embeddings: int,
        features: int,
        embedding_init: Initializer,
        compression_ratio: int = 10,
        *,
        rngs: nnx.Rngs,
    ):
        self.num_hash_embeddings = math.ceil(num_embeddings / compression_ratio)
        self.features = features
        self.embeddings = nnx.Embed(
            num_embeddings=self.num_hash_embeddings,
            features=features,
            embedding_init=embedding_init,
            rngs=rngs,
        )
        self.hash_fn = UniversalHash(
            max_output=self.num_hash_embeddings,
            num_args=1,
            rngs=rngs,
        )
        self.sign_hash_fn = UniversalHash(
            max_output=2,
            num_args=2,
            rngs=rngs,
        )

    def __call__(self, idx):
        embeddings = self.embeddings(self.hash_fn(idx))

        # Scale hash in {0, 1} to {-1, 1}:
        offsets = jnp.arange(self.features)
        signs = 2 * self.sign_hash_fn(idx[..., None], offsets[None, :]) - 1

        return signs * embeddings
