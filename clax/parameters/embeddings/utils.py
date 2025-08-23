import warnings

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from jax import config

EIGHT_MERSENNE_PRIME = 2**31 - 1
NINTH_MERSENNE_PRIME = 2**61 - 1
INT32_MAX = 2**31 - 1


class UniversalHash(nnx.Module):
    """
    A GPU-friendly and lightweight universal hash function
    following Eq. 2 of the ROBE-Z paper.

    References:
    Desai, Li, and Shrivastava (2021). "Random offset block embedding array (robe) for criteotb benchmark mlperf dlrm model..."
    """

    def __init__(
        self,
        max_output: int,
        num_args: int,
        *,
        rngs: nnx.Rngs,
    ):
        super().__init__()

        self.dtype = jnp.int32
        self.large_prime = EIGHT_MERSENNE_PRIME

        if max_output > INT32_MAX:
            self.dtype = np.int64
            self.large_prime = NINTH_MERSENNE_PRIME

            if not config.x64_enabled:
                warnings.warn(
                    f"UniversalHash: max_output ({max_output}) is too large for int32. "
                    "Automatically enabling JAX 64-bit mode (jax_enable_x64=True)."
                )
                config.update("jax_enable_x64", True)

        self.max_output = jnp.asarray(np.array(max_output), dtype=self.dtype)
        self.num_args = num_args

        self.coefficients = jax.random.randint(
            rngs(),
            shape=(self.num_args + 1,),
            minval=1,
            maxval=self.large_prime,
            dtype=self.dtype,
        )
        self.coefficients = self.coefficients.at[0].set(
            jax.random.randint(
                rngs(),
                shape=(),
                minval=0,
                maxval=self.large_prime,
                dtype=self.dtype,
            )
        )

    def __call__(self, *hash_inputs):
        """
        Perform universal hashing in a numerically stable way:
        hash(x) = ((w_0 + w_1 * x_1 + ... + w_n * x_n) % prime) % output
        using additive and multiplicative modulo rules for numerical stability.
        """
        assert (
            len(hash_inputs) == self.num_args
        ), f"UniversalHash expects {self.num_args} arguments, but got {len(hash_inputs)}"

        result = self.coefficients[0].astype(jnp.int64)

        for i, hash_input in enumerate(hash_inputs):
            coeff = self.coefficients[i + 1]
            inp = hash_input.astype(jnp.int64)
            inp = inp % self.large_prime

            term = (coeff * inp) % self.large_prime
            result = (result + term) % self.large_prime

        return (result % self.max_output).astype(self.dtype)
