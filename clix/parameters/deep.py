from typing import Dict

from flax import nnx
from flax.nnx import rnglib
from jax import Array

from clix.models.utils import ActivationFactory
from clix.parameters.base import Parameter


class DeepParameter(Parameter):
    def __init__(
        self,
        use_feature: str,
        features: int,
        hidden_units: int,
        layers: int,
        dropout: float,
        activation: str,
        *,
        rngs: rnglib.Rngs,
    ):
        super().__init__()
        self.use_feature = use_feature
        modules = []
        activation_fn = ActivationFactory[activation]

        for _ in range(layers):
            modules.extend(
                [
                    nnx.Linear(features, hidden_units, rngs=rngs),
                    activation_fn,
                    nnx.Dropout(rate=dropout, rngs=rngs),
                ]
            )
            features = hidden_units

        modules.append(nnx.Linear(features, 1, rngs=rngs))
        self.model = nnx.Sequential(*modules)

    def logit(self, batch: Dict) -> Array:
        return self.model(batch[self.use_feature])

    def prob(self, batch: Dict) -> Array:
        return nnx.sigmoid(self.logit(batch))

    def log_prob(self, batch: Dict) -> Array:
        return nnx.log_sigmoid(self.logit(batch))
