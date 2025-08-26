from dataclasses import dataclass
from typing import Dict, Callable, Optional

from flax import nnx
from flax.nnx import rnglib
from jax import Array

from clax.parameters import ParameterConfig, Parameter


@dataclass
class DeepParameterConfig(ParameterConfig):
    use_feature: str
    features: int
    hidden_units: int = 16
    layers: int = 2
    dropout: float = 0.0
    activation_fn: Callable = nnx.elu
    norm: Optional[Callable] = nnx.LayerNorm
    input_norm: Optional[Callable] = nnx.LayerNorm


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

        self.input_norm = (
            config.input_norm(features, rngs=rngs)
            if config.input_norm is not None
            else lambda x: x
        )

        for _ in range(config.layers):
            modules.append(nnx.Linear(features, config.hidden_units, rngs=rngs))

            if config.norm is not None:
                modules.append(config.norm(config.hidden_units, rngs=rngs))

            modules.extend(
                [
                    config.activation_fn,
                    nnx.Dropout(rate=config.dropout, rngs=rngs),
                ]
            )

            features = config.hidden_units

        modules.append(nnx.Linear(features, 1, rngs=rngs))
        self.model = nnx.Sequential(*modules)

    def logit(self, batch: Dict) -> Array:
        x = self.input_norm(batch[self.config.use_feature])
        return self.model(x).squeeze()

    def prob(self, batch: Dict) -> Array:
        return nnx.sigmoid(self.logit(batch))

    def log_prob(self, batch: Dict) -> Array:
        return nnx.log_sigmoid(self.logit(batch))
