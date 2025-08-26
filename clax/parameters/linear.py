from dataclasses import dataclass
from typing import Dict, Optional, Callable

from flax import nnx
from flax.nnx import rnglib
from jax import Array

from clax.parameters.base import Parameter, ParameterConfig


@dataclass
class LinearParameterConfig(ParameterConfig):
    use_feature: str
    features: int
    input_norm: Optional[Callable] = nnx.LayerNorm


class LinearParameter(Parameter):
    def __init__(
        self,
        config: LinearParameterConfig,
        *,
        rngs: rnglib.Rngs,
    ):
        super().__init__()
        self.config = config
        self.input_norm = (
            config.input_norm(config.features, rngs=rngs)
            if config.input_norm is not None
            else lambda x: x
        )
        self.linear = nnx.Linear(
            in_features=config.features,
            out_features=1,
            rngs=rngs,
        )

    def logit(self, batch: Dict) -> Array:
        x = self.input_norm(batch[self.config.use_feature])
        return self.linear(x).squeeze()

    def prob(self, batch: Dict) -> Array:
        return nnx.sigmoid(self.logit(batch))

    def log_prob(self, batch: Dict) -> Array:
        return nnx.log_sigmoid(self.logit(batch))
