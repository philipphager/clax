import jax
import jax.numpy as jnp
from flax import nnx

EIGHT_MERSENNE_PRIME = 2**31 - 1


class UniversalHash(nnx.Module):
    """
    A GPU-friendly and lightweight universal hash function
    following Eq. 2 of the ROBE-Z paper with comprehensive overflow protection.

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

        self.coefficients = jax.random.randint(
            rngs(),
            shape=(self.num_args + 1,),
            minval=1,
            maxval=large_prime,
            dtype=jnp.int64,
        )
        self.coefficients = self.coefficients.at[0].set(
            jax.random.randint(
                rngs(),
                shape=(),
                minval=0,
                maxval=large_prime,
                dtype=jnp.int64,
            )
        )

    def __call__(self, *hash_inputs):
        assert (
            len(hash_inputs) == self.num_args
        ), f"UniversalHash expects {self.num_args} arguments, but got {len(hash_inputs)}"

        result = self.coefficients[0]

        for i, hash_input in enumerate(hash_inputs):
            input_val = jnp.int64(hash_input)
            term = (self.coefficients[i + 1] * input_val) % self.large_prime
            result = (result + term) % self.large_prime

        return result % self.max_output
