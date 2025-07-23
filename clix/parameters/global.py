from flax import nnx
from flax.nnx.nn import initializers
from flax.typing import Initializer
from jax import Array

from clix.parameters.base import Parameter


class GlobalParameter(Parameter):
    def __init__(
        self,
        parameters: int = 1,
        initializers: Initializer = initializers.normal(0.5),
        *,
        rngs: nnx.Rngs,
    ):
        super().__init__()
        self.weight = nnx.Param(initializers(rngs.params(), parameters))

    def logit(self, *args, **kwargs) -> Array:
        return self.weight.value

    def prob(self, *args, **kwargs) -> Array:
        return nnx.sigmoid(self.logit())

    def log_prob(self, *args, **kwargs) -> Array:
        return nnx.log_sigmoid(self.logit())
