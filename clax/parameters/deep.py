from dataclasses import dataclass
from typing import Dict, Callable

from clax.parameters import ParameterConfig, Parameter
from flax import nnx
from flax.nnx import rnglib
from jax import Array


@dataclass
class DeepParameterConfig(ParameterConfig):
    use_feature: str
    features: int
    hidden_units: int = 16
    layers: int = 2
    dropout: float = 0.0
    activation_fn: Callable = nnx.elu


class DeepParameter(Parameter):
    """
    Parameter using input features and a deep feed forward network, e.g.,
    to model user attraction from query-document features.
    """

    def __init__(
        self,
        config: DeepParameterConfig,
        *,
        rngs: rnglib.Rngs,
    ):
        super().__init__()
        self.config = config
        modules = []
        features = config.features

        for _ in range(config.layers):
            modules.extend(
                [
                    nnx.Linear(features, config.hidden_units, rngs=rngs),
                    config.activation_fn,
                    nnx.Dropout(rate=config.dropout, rngs=rngs),
                ]
            )
            features = config.hidden_units

        modules.append(nnx.Linear(features, 1, rngs=rngs))
        self.model = nnx.Sequential(*modules)

    def logit(self, batch: Dict) -> Array:
        return self.model(batch[self.config.use_feature]).squeeze()

    def prob(self, batch: Dict) -> Array:
        return nnx.sigmoid(self.logit(batch))

    def log_prob(self, batch: Dict) -> Array:
        return nnx.log_sigmoid(self.logit(batch))
