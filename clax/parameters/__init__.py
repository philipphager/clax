from flax import nnx

from .base import GlobalParameter, GlobalParameterConfig
from .base import Parameter, ParameterConfig
from .deep import DeepParameter, DeepParameterConfig
from .embeddings import (
    EmbeddingParameter,
    EmbeddingParameterConfig,
    FullEmbedding,
    HashEmbedding,
    QREmbedding,
    RobeDEmbedding,
)
from .linear import LinearParameter, LinearParameterConfig


def build_parameter(config: ParameterConfig, rngs: nnx.Rngs) -> Parameter:
    if isinstance(config, GlobalParameterConfig):
        return GlobalParameter(config, rngs=rngs)
    elif isinstance(config, EmbeddingParameterConfig):
        return EmbeddingParameter(config, rngs=rngs)
    elif isinstance(config, LinearParameterConfig):
        return LinearParameter(config, rngs=rngs)
    elif isinstance(config, DeepParameterConfig):
        return DeepParameter(config, rngs=rngs)
    else:
        raise ValueError(f"Unknown parameter config type: {type(config)}")
