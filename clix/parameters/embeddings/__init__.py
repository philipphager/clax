from .base import EmbeddingParameter, EmbeddingParameterConfig, FullEmbedding
from .compositional import QREmbedding
from .hashing import HashEmbedding
from .robe import RobeDEmbedding


class EmbeddingFactory:
    """
    Global factory to retrieve embedding methods by name.
    """

    name2embedding = {
        "full": FullEmbedding,
        "hashing": HashEmbedding,
        "qr": QREmbedding,
        "robe-d": RobeDEmbedding,
    }

    @classmethod
    def get(cls, name):
        if name not in cls.name2embedding:
            raise Exception(
                f"Unknown embedding method {name}, "
                f"must be one of {list(cls.name2embedding.keys())}"
            )

        return cls.name2embedding[name]

    @classmethod
    def register(cls, name, activation_fn):
        if name in cls.name2embedding:
            raise Exception(f"Embedding {name} already registered")
        cls.name2embedding[name] = activation_fn
