import jax.numpy as jnp
from flax import nnx
from jax import Array


def logits_to_log_probs(logits: Array) -> Array:
    """
    Computes log(sigmoid(x)) from logits in a numerically stable way.
    """
    return nnx.log_sigmoid(logits)


def logits_to_complement_log_probs(logits: Array) -> Array:
    """
    Computes log(1-sigmoid(x)) = log_sigmoid(-x) from logits.
    """
    return nnx.log_sigmoid(-logits)


def exp_logp(log_probs: Array, *, where: Array) -> Array:
    return jnp.where(where, jnp.exp(log_probs), 0.0)
