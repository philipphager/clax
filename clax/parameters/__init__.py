from typing import Optional, Callable, Dict, Union

from flax import nnx

from .base import Parameter, ParameterConfig
from .deep import DeepParameter, DeepParameterConfig
from .deep_cross import DeepCrossParameter, DeepCrossParameterConfig
from .embeddings import (
    EmbeddingParameter,
    EmbeddingParameterConfig,
    FullEmbedding,
    HashEmbedding,
    QREmbedding,
    RobeDEmbedding,
)
from .globals import GlobalParameter, GlobalParameterConfig
from .linear import LinearParameter, LinearParameterConfig


def init_parameter(
    name: str,
    parameter_or_config: Optional[Union[Parameter, ParameterConfig]],
    default_config_fn: Callable,
    default_config_args: Dict,
    rngs: nnx.Rngs,
) -> Parameter:
    """
    Initializes a model parameter. If no config or parameter instance is provided,
    we instantiate a default parameter. Otherwise, we instantiate a parameter from its
    config or pass a provided parameter instance through.
    """
    if parameter_or_config is None:
        parameter_or_config = default_config_fn(**default_config_args)

    if isinstance(parameter_or_config, ParameterConfig):
        return build_parameter(parameter_or_config, rngs)
    elif isinstance(parameter_or_config, Parameter):
        return parameter_or_config
    else:
        raise TypeError(
            f"'{name}' must be an instance of Parameter or ParameterConfig, "
            f"but got type {type(parameter_or_config).__name__}."
        )


def build_parameter(config: ParameterConfig, rngs: nnx.Rngs) -> Parameter:
    if isinstance(config, GlobalParameterConfig):
        return GlobalParameter(config, rngs=rngs)
    elif isinstance(config, EmbeddingParameterConfig):
        return EmbeddingParameter(config, rngs=rngs)
    elif isinstance(config, LinearParameterConfig):
        return LinearParameter(config, rngs=rngs)
    elif isinstance(config, DeepParameterConfig):
        return DeepParameter(config, rngs=rngs)
    elif isinstance(config, DeepCrossParameterConfig):
        return DeepCrossParameter(config, rngs=rngs)
    else:
        raise ValueError(f"Unknown parameter config type: {type(config)}")
