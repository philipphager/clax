import jax.numpy as jnp
import pytest
from jax import Array

from clax.models.utils import last_clicked_positions


@pytest.mark.parametrize(
    "positions, clicks, expected_positions",
    [
        (
            jnp.array([[1, 2, 3, 4, 5]]),
            jnp.array([[0, 0, 0, 0, 0]]),
            jnp.array([[0, 0, 0, 0, 0]]),
        ),
        (
            jnp.array([[1, 2, 3, 4, 5]]),
            jnp.array([[1, 0, 0, 0, 0]]),
            jnp.array([[0, 1, 1, 1, 1]]),
        ),
        (
            jnp.array([[1, 2, 3, 4, 5]]),
            jnp.array([[1, 0, 0, 0, 1]]),
            jnp.array([[0, 1, 1, 1, 1]]),
        ),
        (
            jnp.array([[1, 2, 3, 4, 5]]),
            jnp.array([[1, 0, 1, 0, 1]]),
            jnp.array([[0, 1, 1, 3, 3]]),
        ),
        (
            jnp.array([[1, 2, 3, 4, 5]]),
            jnp.array([[1, 1, 1, 1, 1]]),
            jnp.array([[0, 1, 2, 3, 4]]),
        ),
        (
            jnp.array([[1, 3, 4, 5, 6]]),
            jnp.array([[1, 1, 1, 1, 1]]),
            jnp.array([[0, 1, 3, 4, 5]]),
        ),
        (
            jnp.array([[1, 2, 3, 4, 5], [10, 11, 12, 13, 14]]),
            jnp.array([[1, 0, 1, 1, 0], [0, 1, 0, 0, 1]]),
            jnp.array([[0, 1, 1, 3, 4], [0, 0, 11, 11, 11]]),
        ),
    ],
)
def test_last_clicked_positions(
    positions: Array,
    clicks: Array,
    expected_positions: Array,
):
    assert jnp.array_equal(
        last_clicked_positions(positions, clicks),
        expected_positions,
    )
