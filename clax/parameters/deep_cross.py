from dataclasses import dataclass
from enum import Enum
from typing import Dict, Callable, Optional

import jax.numpy as jnp
from flax import nnx
from flax.nnx import rnglib
from jax import Array

from clax.parameters import ParameterConfig, Parameter


class Combination(str, Enum):
    STACKED = "stacked"
    PARALLEL = "parallel"


@dataclass
class DeepCrossParameterConfig(ParameterConfig):
    use_feature: str
    features: int
    hidden_units: int = 32
    dropout: float = 0.0
    cross_layers: int = 2
    deep_layers: int = 2
    activation_fn: Callable = nnx.elu
    norm: Optional[Callable] = nnx.LayerNorm
    input_norm: Optional[Callable] = nnx.LayerNorm
    combination: Combination = Combination.STACKED


class DeepCrossParameter(Parameter):
    def __init__(
        self,
        config: DeepCrossParameterConfig,
        *,
        rngs: rnglib.Rngs,
    ):
        super().__init__()
        self.config = config
        self.input_norm = (
            config.input_norm(config.features, rngs=rngs)
            if config.input_norm is not None
            else lambda x: x
        )
        self.deep_model = self._get_deep_model(config, rngs)
        self.cross_modules = self._get_cross_modules(config, rngs)

        logit_units = (
            config.hidden_units
            if config.combination == Combination.STACKED
            else config.hidden_units + config.features
        )

        self.logit_layer = nnx.Linear(logit_units, 1, rngs=rngs)

    def _get_deep_model(self, config: DeepCrossParameterConfig, rngs):
        modules = []
        features = config.features

        for _ in range(config.deep_layers):
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

        return nnx.Sequential(*modules)

    def _get_cross_modules(self, config: DeepCrossParameterConfig, rngs):
        modules = []
        features = config.features

        for _ in range(config.cross_layers):
            modules.append(CrossLayer(features, rngs=rngs))

        return modules

    def _forward_cross(self, x):
        x0 = x

        for module in self.cross_modules:
            x = module(x0, x)

        return x

    def logit(self, batch: Dict) -> Array:
        x = self.input_norm(batch[self.config.use_feature])

        x_cross = self._forward_cross(x)
        x_deep = x_cross if self.config.combination == Combination.STACKED else x
        x_deep = self.deep_model(x_deep)
        x = (
            x_deep
            if self.config.combination == Combination.STACKED
            else jnp.concatenate([x_deep, x_cross], axis=-1)
        )
        return self.logit_layer(x).squeeze()

    def prob(self, batch: Dict) -> Array:
        return nnx.sigmoid(self.logit(batch))

    def log_prob(self, batch: Dict) -> Array:
        return nnx.log_sigmoid(self.logit(batch))


class CrossLayer(nnx.Module):
    def __init__(
        self,
        features: int,
        *,
        rngs: rnglib.Rngs,
    ):
        super().__init__()
        self.linear = nnx.Linear(features, features, rngs=rngs)

    def __call__(self, x0: Array, x: Array):
        # Feature crossing with original input + residual connection:
        return x0 * self.linear(x) + x
