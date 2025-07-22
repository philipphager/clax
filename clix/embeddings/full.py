from dataclasses import dataclass

from flax import nnx

from clix.embeddings.base import EmbeddingConfig


@dataclass
class FullEmbeddingConfig(EmbeddingConfig):

    def create_embedding(self, num_embeddings: int, rngs: nnx.Rngs) -> nnx.Module:
        return nnx.Embed(
            num_embeddings=num_embeddings,
            features=self.features,
            embedding_init=self.embedding_init,
            rngs=rngs,
        )
