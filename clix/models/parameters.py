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
        self.weight = nnx.Param(initializers(rngs.params(), shape))

    def __call__(self) -> Array:
        return nnx.sigmoid(self.weight.value)

    def logit(self) -> Array:
        return self.weight.value

    def prob(self) -> Array:
        return nnx.sigmoid(self.weight.value)

    def log_prob(self) -> Array:
        return nnx.log_sigmoid(self.weight.value)


class BernoulliEmbedding(nnx.Module):
    def __init__(
        self,
        use_feature: str,
        parameters: int,
        add_baseline: bool = True,
        *,
        rngs: nnx.Rngs,
    ):
        super().__init__()
        self.use_feature = use_feature
        self.add_baseline = add_baseline
        self.baseline = nnx.Param(jnp.zeros(1))
        self.embeddings = nnx.Embed(
            num_embeddings=parameters,
            features=1,
            rngs=rngs,
            embedding_init=initializers.zeros_init(),
        )

    def logit(self, batch: Dict) -> Array:
        x = batch[self.use_feature]
        logit = self.embeddings(x).squeeze()

        if self.add_baseline:
            # Add a baseline prediction, similar to a wide&deep model.
            # The model resorts to avg. predictions for prev. unseen parameters:
            logit = self.baseline.value + logit

        return logit

    def prob(self, batch: Dict) -> Array:
        return nnx.sigmoid(self.logit(batch))

    def log_prob(self, batch: Dict) -> Array:
        return nnx.log_sigmoid(self.logit(batch))


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
