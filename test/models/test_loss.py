import jax.numpy as jnp
import pytest
from jax import Array

from clix.models.loss import binary_cross_entropy


@pytest.mark.parametrize(
    "y_predict, y_true, mask, expected_loss",
    [
        (
            jnp.array([[0.1, 0.5, 0.9]]),
            jnp.array([[0, 1, 1]]),
            jnp.array([[True, True, True]]),
            jnp.array([[-(jnp.log(1 - 0.1) + jnp.log(0.5) + jnp.log(0.9)) / 3]]),
        ),
        (
            jnp.array([[0.1, 0.5, 0.9]]),
            jnp.array([[0, 1, 1]]),
            jnp.array([[True, True, False]]),
            jnp.array([[-(jnp.log(1 - 0.1) + jnp.log(0.5)) / 2]]),
        ),
        (
            jnp.array([[0.1, 0.5, 0.9], [0.2, 0.4, 0.8]]),
            jnp.array([[0, 1, 1], [1, 0, 0]]),
            jnp.array([[True, True, True], [True, True, True]]),
            jnp.array(
                [
                    -(jnp.log(1 - 0.1) + jnp.log(0.5) + jnp.log(0.9)) / 3,
                    -(jnp.log(0.2) + jnp.log(1 - 0.4) + jnp.log(1 - 0.8)) / 3,
                ]
            ),
        ),
    ],
)
def test_binary_cross_entropy(
    y_predict: Array,
    y_true: Array,
    mask: Array,
    expected_loss: Array,
):
    assert jnp.allclose(
        binary_cross_entropy(y_predict, y_true, mask),
        expected_loss,
    )
