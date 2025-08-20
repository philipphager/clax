from dataclasses import dataclass
from typing import Dict, Callable

from flax import nnx
from flax.nnx.nn import initializers
from flax.typing import Initializer
from jax import Array

from clax.parameters import Parameter, ParameterConfig

# Alias NNX embedding layer for clarity:
FullEmbedding = nnx.Embed

near_zero_init = initializers.variance_scaling(
    1e-05,
    "fan_in",
    "normal",
    out_axis=0,
)


@dataclass
class EmbeddingParameterConfig(ParameterConfig):
    use_feature: str
    parameters: int
    embedding_features: int = 1
    add_baseline: bool = True
    has_padding: bool = True
    embedding_fn: Callable = FullEmbedding
    baseline_init: Initializer = initializers.ones
    embedding_init: Initializer = near_zero_init


class EmbeddingParameter(Parameter):
    def __init__(
        self,
        config: EmbeddingParameterConfig,
        *,
        rngs: nnx.Rngs,
    ):
        super().__init__()
        self.config = config

        # Make sure to allocate an additional parameter if input uses padding:
        num_embeddings = (
            config.parameters + 1 if config.has_padding else config.parameters
        )
        self.baseline = nnx.Param(config.baseline_init(rngs.params(), (1,)))
        self.embeddings = config.embedding_fn(
            num_embeddings=num_embeddings,
            features=config.embedding_features,
            embedding_init=config.embedding_init,
            rngs=rngs,
        )

        self.projection = nnx.Linear(config.embedding_features, 1, rngs=rngs)
        self.add_projection = config.embedding_features > 1
        self.add_baseline = self.config.add_baseline

    def logit(self, batch: Dict) -> Array:
        x = batch[self.config.use_feature]
        logit = self.embeddings(x)

        if self.add_projection:
            logit = self.projection(logit)

        if self.add_baseline:
            logit = self.baseline.value + logit

        return logit.squeeze()

    def prob(self, batch: Dict) -> Array:
        return nnx.sigmoid(self.logit(batch))

    def log_prob(self, batch: Dict) -> Array:
        return nnx.log_sigmoid(self.logit(batch))
