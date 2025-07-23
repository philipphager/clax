from abc import ABC, abstractmethod
from typing import Dict

from flax import nnx
from flax.nnx.nn import initializers
from flax.typing import Initializer
from jax import Array


class Parameter(nnx.Module, ABC):
    @abstractmethod
    def logit(self, batch) -> Array:
        pass

    @abstractmethod
    def prob(self, batch: Dict) -> Array:
        pass

    @abstractmethod
    def log_prob(self, batch: Dict) -> Array:
        pass


class GlobalParameter(Parameter):
    """
    Unconditional, global parameter that does not depend on any input features.
    E.g., to model continuation in the DBN model.
    """

    def __init__(
        self,
        parameters: int = 1,
        initializers: Initializer = initializers.normal(0.5),
        *,
        rngs: nnx.Rngs,
    ):
        super().__init__()
        self.weight = nnx.Param(initializers(rngs.params(), (parameters,)))

    def logit(self, *args, **kwargs) -> Array:
        return self.weight.value

    def prob(self, *args, **kwargs) -> Array:
        return nnx.sigmoid(self.logit())

    def log_prob(self, *args, **kwargs) -> Array:
        return nnx.log_sigmoid(self.logit())
