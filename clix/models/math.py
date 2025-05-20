import jax.numpy as jnp

from jax import Array

def exp_logp(log_probs: Array, *, where: Array) -> Array:
    return jnp.where(where, jnp.exp(log_probs), 0.0)
