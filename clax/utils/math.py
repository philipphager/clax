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


def probs_to_log_probs(probs: Array, eps=1e-10) -> Array:
    return jnp.log(jnp.clip(probs, eps, 1.0))


def log1mexp(x):
    """
    Numerically stable calculation of log(1 - exp(x)).

    References:
    [1]: Machler, Martin. Accurately computing log(1 - exp(-|a|))
    https://cran.r-project.org/web/packages/Rmpfr/vignettes/log1mexp-note.pdf
    [2] Tensorflow Probability
    https://github.com/tensorflow/probability/blob/65f265c62bb1e2d15ef3e25104afb245a6d52429/tensorflow_probability/python/math/generic.py#L685-L709
    """
    x = -jnp.abs(x)  # Force x <= 0

    # Switching point at x = -log(2) ≈ -0.693 recommended in [1]
    condition = x > -jnp.log(2.0)

    # For x > -log(2): use log(-expm1(x))
    # For x <= -log(2): use log1p(-exp(x))
    return jnp.where(
        condition,
        jnp.log(-jnp.expm1(x)),  # x close to 0
        jnp.log1p(-jnp.exp(x)),  # x very negative
    )
