from abc import ABC, abstractmethod
from typing import Dict

from flax import nnx
from jax import Array


class ParameterConfig(ABC):
    pass


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
