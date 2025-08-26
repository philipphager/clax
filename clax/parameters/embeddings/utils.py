import warnings

import jax
import jax.numpy as jnp
from flax import nnx

EIGHT_MERSENNE_PRIME = 2**31 - 1
INT32_MAX_VALUE = 2**31 - 1


class UniversalHash(nnx.Module):
    """
    A GPU-friendly and lightweight universal hash function
    following Eq. 2 of the ROBE-Z paper and adding linear scaling to avoid modulo bias.

    References:
    Desai, Li, and Shrivastava (2021). "Random offset block embedding array (robe) for criteotb benchmark mlperf dlrm model..."
    """

    def __init__(
        self,
        max_output: int,
        num_args: int,
        *,
        rngs: nnx.Rngs,
        large_prime: int = EIGHT_MERSENNE_PRIME,
    ):
        super().__init__()
        self.large_prime = jnp.int64(large_prime)
        self.max_output = jnp.int64(max_output)
        self.num_args = num_args

        if not jax.config.x64_enabled:
            warnings.warn(
                "UniversalHash intermediary outputs can exceed int32. "
                "Automatically enabling JAX 64-bit mode (jax_enable_x64=True)."
            )
            jax.config.update("jax_enable_x64", True)

        self.coefficients = jax.random.randint(
            rngs(),
            shape=(self.num_args + 1,),
            minval=1,
            maxval=self.large_prime,
            dtype=jnp.int64,
        )
        self.coefficients = self.coefficients.at[0].set(
            jax.random.randint(
                rngs(),
                shape=(),
                minval=0,
                maxval=self.large_prime,
                dtype=jnp.int64,
            )
        )

    def __call__(self, *hash_inputs):
        """
        Perform universal hashing:
        hash(x) = ((w_0 + w_1 * x_1 + ... + w_n * x_n) % prime) % output
        We replace the final % output with linear scaling to avoid modulo bias.
        """

        assert len(hash_inputs) == self.num_args, (
            f"UniversalHash expects {self.num_args} arguments, but got {len(hash_inputs)}"
        )

        # Start with constant term in range [0, prime):
        result = self.coefficients[0]

        for i, hash_input in enumerate(hash_inputs):
            hash_input = hash_input.astype(dtype=jnp.int64)

            # Multiply a term in range [1, prime) to each provided input:
            result += self.coefficients[i + 1] * hash_input

        hash_value = result % self.large_prime

        # Linearly scale value to output size to avoid modulo bias:
        return jnp.floor(
            hash_value.astype(jnp.float64)
            / self.large_prime.astype(jnp.float64)
            * self.max_output
        ).astype(jnp.int64)
