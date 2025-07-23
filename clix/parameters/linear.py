from typing import Dict

from flax import nnx
from flax.nnx import rnglib
from jax import Array

from clix.parameters.base import Parameter


class LinearParameter(Parameter):
    def __init__(
        self,
        use_feature: str,
        features: int,
        *,
        rngs: rnglib.Rngs,
    ):
        super().__init__()
        self.use_feature = use_feature
        self.linear = nnx.Linear(in_features=features, out_features=1, rngs=rngs)

    def logit(self, batch: Dict) -> Array:
        return self.linear(batch[self.use_feature]).squeeze()

    def prob(self, batch: Dict) -> Array:
        return nnx.sigmoid(self.logit(batch))

    def log_prob(self, batch: Dict) -> Array:
        return nnx.log_sigmoid(self.logit(batch))
