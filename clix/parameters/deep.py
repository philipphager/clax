from typing import Dict

from flax import nnx
from flax.nnx import rnglib
from jax import Array


class ActivationFactory:
    """
    Global factory to retrieve activation functions by name.
    Can be used to register custom activation functions:

    >>> ActivationFactory.register("selu", nnx.selu)
    """

    name2activation = {
        "tanh": nnx.tanh,
        "relu": nnx.relu,
        "elu": nnx.elu,
        "gelu": nnx.gelu,
    }

    @classmethod
    def get(cls, name):
        return cls.name2activation[name]

    @classmethod
    def register(cls, name, activation_fn):
        if name in cls.name2activation:
            raise Exception(f"Activation {name} already registered")
        cls.name2activation[name] = activation_fn


class DeepParameter(nnx.Module):
    """
    Parameter using input features and a deep feed forward network, e.g.,
    to model user attraction from query-document features.
    """

    def __init__(
        self,
        use_feature: str,
        features: int,
        hidden_units: int = 16,
        layers: int = 2,
        dropout: float = 0.0,
        activation: str = "elu",
        *,
        rngs: rnglib.Rngs,
    ):
        super().__init__()
        self.use_feature = use_feature
        modules = []
        activation_fn = ActivationFactory.get(activation)

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
        return self.model(batch[self.use_feature]).squeeze()

    def prob(self, batch: Dict) -> Array:
        return nnx.sigmoid(self.logit(batch))

    def log_prob(self, batch: Dict) -> Array:
        return nnx.log_sigmoid(self.logit(batch))
