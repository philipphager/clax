from dataclasses import dataclass

from flax import nnx
from flax.nnx.nn import initializers
from flax.typing import Initializer
from jax import Array

from clax.parameters import ParameterConfig, Parameter


@dataclass
class GlobalParameterConfig(ParameterConfig):
    parameters: int = 1
    initializers: Initializer = initializers.normal(0.5)


class GlobalParameter(Parameter):
    """
    Unconditional, global parameter that does not depend on any input features.
    E.g., to model continuation in the DBN model.
    """

    def __init__(
        self,
        config: GlobalParameterConfig = GlobalParameterConfig(),
        *,
        rngs: nnx.Rngs,
    ):
        super().__init__()
        self.config = config
        self.weight = nnx.Param(
            config.initializers(rngs.params(), (config.parameters,))
        )

    def logit(self, *args, **kwargs) -> Array:
        return self.weight.value

    def prob(self, *args, **kwargs) -> Array:
        return nnx.sigmoid(self.logit())

    def log_prob(self, *args, **kwargs) -> Array:
        return nnx.log_sigmoid(self.logit())
