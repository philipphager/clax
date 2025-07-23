from typing import Dict, Callable

from flax import nnx
from flax.nnx.nn import initializers
from flax.typing import Initializer
from jax import Array

from clix.parameters import Parameter

# Alias NNX embedding layer for clarity:
FullEmbedding = nnx.Embed

near_zero_init = initializers.variance_scaling(
    1e-05,
    "fan_in",
    "normal",
    out_axis=0,
)


class EmbeddingParameter(Parameter):
    def __init__(
        self,
        use_feature: str,
        parameters: int,
        embedding_features: int = 4,
        add_baseline: bool = True,
        embedding_fn: Callable = FullEmbedding,
        baseline_init: Initializer = initializers.ones,
        embedding_init: Initializer = near_zero_init,
        *,
        rngs: nnx.Rngs,
        **kwargs,
    ):
        super().__init__()
        self.use_feature = use_feature
        self.add_baseline = add_baseline
        self.baseline = nnx.Param(baseline_init(rngs.params(), (1,)))
        self.embeddings = embedding_fn(
            num_embeddings=parameters,
            features=embedding_features,
            embedding_init=embedding_init,
            rngs=rngs,
            **kwargs,
        )
        self.projection = nnx.Linear(embedding_features, 1, rngs=rngs)

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
