import jax.numpy as jnp
from jax import Array
from jax import lax


def last_clicked_positions(positions: Array, clicks: Array) -> Array:
    # Filter clicked positions, e.g.: [1, 2, 3, 4], [1, 0, 0, 1] -> [1, 0, 0, 4]
    clicked_positions = jnp.where(clicks == 1, positions, 0)
    # Find the last clicked position for each item: [1, 0, 0, 4] -> [1, 1, 1, 4]
    # Assumes positions are sorted in ascending order!
    clicked_positions = lax.cummax(clicked_positions, axis=1)
    # Shift the clicked positions to the right to align with the next item:
    clicked_positions = jnp.roll(clicked_positions, shift=1, axis=1)
    # Set the first position to 0, as there is no previously clicked position:
    return clicked_positions.at[:, 0].set(0)
