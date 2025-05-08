from typing import Dict

from flax import nnx
from jax import Array


class BernoulliEmbedding(nnx.Module):
    def __init__(
        self,
        use_feature: str,
        parameters: int,
        *,
        rngs: nnx.Rngs,
    ):
        super().__init__()
        parameters = parameters + 1 # Embedding ids are 1-indexed as zero is padding.
        self.use_feature = use_feature
        self.embeddings = nnx.Embed(num_embeddings=parameters, features=1, rngs=rngs)

    def __call__(self, batch: Dict) -> Array:
        x = batch[self.use_feature]
        return nnx.sigmoid(self.embeddings(x).squeeze())


class BetaEmbedding(nnx.Module):
    def __init__(
        self,
        use_feature: str,
        parameters: int,
        *,
        rngs: nnx.Rngs,
    ):
        super().__init__()
        parameters = parameters + 1 # Embedding ids are 1-indexed as zero is padding.
        self.use_feature = use_feature
        self.alpha = nnx.Sequential(nnx.Embed(num_embeddings=parameters, features=1, rngs=rngs), nnx.softplus, self._offset)
        self.beta = nnx.Sequential(nnx.Embed(num_embeddings=parameters, features=1, rngs=rngs), nnx.softplus, self._offset)

    def __call__(self, batch: Dict) -> Array:
        x = batch[self.use_feature]
        alpha = self.alpha(x).squeeze()
        beta = self.beta(x).squeeze()
        return alpha / (alpha + beta)

    @staticmethod
    def _offset(x):
        return x + 2
