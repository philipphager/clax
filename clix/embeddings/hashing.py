import math

from flax import nnx
from flax.typing import Initializer


class HashEmbedding(nnx.Module):
    def __init__(
        self,
        num_embeddings: int,
        features: int,
        embedding_init: Initializer,
        *,
        num_collisions: int = 4,
        rngs: nnx.Rngs,
    ):
        self.num_hash_embeddings = math.ceil(num_embeddings / num_collisions)
        self.embeddings = nnx.Embed(
            num_embeddings=self.num_hash_embeddings,
            features=features,
            embedding_init=embedding_init,
            rngs=rngs,
        )

    def __call__(self, idx):
        return self.embeddings(idx % self.num_hash_embeddings)
