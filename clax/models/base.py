from abc import ABC, abstractmethod
from typing import Dict

from flax import nnx
from jax import Array


class ClickModel(nnx.Module, ABC):
    @abstractmethod
    def compute_loss(self, batch: Dict, aggregate: bool = True):
        pass

    @abstractmethod
    def predict_conditional_clicks(self, batch: Dict) -> Array:
        pass

    @abstractmethod
    def predict_clicks(self, batch: Dict) -> Array:
        pass

    @abstractmethod
    def predict_relevance(self, batch: Dict) -> Array:
        pass

    @abstractmethod
    def sample(self, batch: Dict, rngs: nnx.Rngs):
        pass
