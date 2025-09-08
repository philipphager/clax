import jax.numpy as jnp
from jax import Array

from clax.utils.math import log1mexp


def binary_cross_entropy(
    y_predict: Array,
    y_true: Array,
    where: Array,
    log_probs: bool = True,
    aggregate: bool = True,
):
    if log_probs:
        p_click = y_predict
        p_no_click = log1mexp(y_predict)
    else:
        p_click = jnp.log(y_predict)
        p_no_click = jnp.log1p(-1.0 * y_predict)

    loss = -y_true * p_click - (1 - y_true) * p_no_click

    if aggregate:
        return loss.mean(where=where, axis=-1)
    else:
        loss = jnp.where(where, loss, 0)
        return loss
