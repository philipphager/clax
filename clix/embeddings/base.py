from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from flax import nnx
from flax.typing import Initializer
from flax.nnx.nn import initializers


@dataclass
class EmbeddingConfig(ABC):
    features: int = 4
    embedding_init: Initializer = field(
        default_factory=lambda: initializers.variance_scaling(
            1e-05, "fan_in", "normal", out_axis=0
        )
    )

    @abstractmethod
    def create_embedding(self, num_embeddings: int, rngs: nnx.Rngs) -> nnx.Module:
        pass
