from typing import Dict, Tuple

import jax.numpy as jnp
from flax import nnx
from flax.nnx import initializers, Initializer
from jax import Array


class BernoulliParameter(nnx.Module):
    def __init__(
        self,
        shape: Tuple[int] = (1,),
        initializers: Initializer = initializers.normal(0.5),
        *,
        rngs: nnx.Rngs,
    ):
        super().__init__()
        self.weight = nnx.Param(initializers(rngs.params(), shape, dtype=jnp.float32))

    def __call__(self) -> Array:
        return nnx.sigmoid(self.weight.value)


class BernoulliEmbedding(nnx.Module):
    def __init__(
        self,
        use_feature: str,
        parameters: int,
        *,
        rngs: nnx.Rngs,
    ):
        super().__init__()
        self.use_feature = use_feature
        self.embeddings = nnx.Embed(num_embeddings=parameters, features=1, rngs=rngs)

    def __call__(self, batch: Dict) -> Array:
        x = batch[self.use_feature]
        return nnx.sigmoid(self.embeddings(x).squeeze())


class BernoulliEmbedding(nnx.Module):
    def __init__(
        self,
        use_feature: str,
        parameters: int,
        *,
        log_prob: bool = False,
        rngs: nnx.Rngs,
    ):
        super().__init__()
        self.use_feature = use_feature
        self.embeddings = nnx.Embed(num_embeddings=parameters, features=1, rngs=rngs)
        self.activation_fn = nnx.log_sigmoid if log_prob else nnx.sigmoid

    def __call__(self, batch: Dict) -> Array:
        x = batch[self.use_feature]
        return self.activation_fn(self.embeddings(x).squeeze())


class BetaEmbedding(nnx.Module):
    def __init__(
        self,
        use_feature: str,
        parameters: int,
        *,
        rngs: nnx.Rngs,
    ):
        super().__init__()
        self.use_feature = use_feature
        self.alpha = nnx.Sequential(
            nnx.Embed(num_embeddings=parameters, features=1, rngs=rngs),
            nnx.softplus,
            self._offset,
        )
        self.beta = nnx.Sequential(
            nnx.Embed(num_embeddings=parameters, features=1, rngs=rngs),
            nnx.softplus,
            self._offset,
        )

    def __call__(self, batch: Dict) -> Array:
        x = batch[self.use_feature]
        alpha = self.alpha(x).squeeze()
        beta = self.beta(x).squeeze()
        return alpha / (alpha + beta)

    @staticmethod
    def _offset(x):
        return x + 2
