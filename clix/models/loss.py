import jax.numpy as jnp

from jax import Array


def binary_cross_entropy(y_predict: Array, y_true: Array, where: Array):
    p_click = jnp.log(y_predict)
    p_no_click = jnp.log1p(-1.0 * y_predict)
    loss = -y_true * p_click - (1 - y_true) * p_no_click

    return loss.mean(where=where, axis=-1)
