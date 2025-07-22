from typing import Dict, Tuple

import jax.numpy as jnp
from flax import nnx
from flax.nnx import Initializer
from flax.nnx.nn import initializers
from jax import Array

from clix.embeddings.base import EmbeddingConfig
from clix.embeddings.full import FullEmbeddingConfig


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
        embedding_config: EmbeddingConfig = FullEmbeddingConfig(),
        *,
        rngs: nnx.Rngs,
    ):
        super().__init__()
        self.use_feature = use_feature
        self.add_baseline = add_baseline
        self.baseline = nnx.Param(jnp.zeros(1))
        self.embeddings = embedding_config.create_embedding(parameters, rngs)
        self.projection = nnx.Linear(4, 1, rngs=rngs)

    def logit(self, batch: Dict) -> Array:
        x = batch[self.use_feature]
        logit = self.projection(self.embeddings(x)).squeeze()

        if self.add_baseline:
            # Add a baseline prediction, similar to a wide&deep model.
            # The model resorts to avg. predictions for prev. unseen parameters:
            logit = self.baseline.value + logit

        return logit

    def prob(self, batch: Dict) -> Array:
        return nnx.sigmoid(self.logit(batch))

    def log_prob(self, batch: Dict) -> Array:
        return nnx.log_sigmoid(self.logit(batch))
