from dataclasses import dataclass
from typing import Dict

from flax import nnx
from flax.nnx import rnglib
from jax import Array

from clax.parameters.base import Parameter, ParameterConfig


@dataclass
class LinearParameterConfig(ParameterConfig):
    use_feature: str
    features: int


class LinearParameter(Parameter):
    def __init__(
        self,
        config: LinearParameterConfig,
        *,
        rngs: rnglib.Rngs,
    ):
        super().__init__()
        self.config = config
        self.linear = nnx.Linear(
            in_features=config.features,
            out_features=1,
            rngs=rngs,
        )

    def logit(self, batch: Dict) -> Array:
        return self.linear(batch[self.config.use_feature]).squeeze()

    def prob(self, batch: Dict) -> Array:
        return nnx.sigmoid(self.logit(batch))

    def log_prob(self, batch: Dict) -> Array:
        return nnx.log_sigmoid(self.logit(batch))
