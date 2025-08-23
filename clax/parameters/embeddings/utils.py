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
        # Store everything as int64 to handle large values (3B+ embeddings)
        # Use numpy first to avoid JAX's int32 coercion during conversion
        import numpy as np

        self.large_prime = jnp.asarray(np.int64(large_prime))
        self.max_output = jnp.asarray(np.int64(max_output))
        self.num_args = num_args

        # Generate coefficients as int64 from the start
        self.coefficients = jax.random.randint(
            rngs(),
            shape=(self.num_args + 1,),
            minval=1,
            maxval=large_prime,
            dtype=jnp.int64,  # Explicitly use int64
        )
        self.coefficients = self.coefficients.at[0].set(
            jax.random.randint(
                rngs(),
                shape=(),
                minval=0,
                maxval=large_prime,
                dtype=jnp.int64,  # Explicitly use int64
            )
        )

    def __call__(self, *hash_inputs):
        assert (
            len(hash_inputs) == self.num_args
        ), f"UniversalHash expects {self.num_args} arguments, but got {len(hash_inputs)}"

        # Start with constant term, already int64 from initialization
        result = self.coefficients[0]

        for i, hash_input in enumerate(hash_inputs):
            # Cast input to int64 and apply modular arithmetic to prevent overflow
            input_val = jnp.int64(hash_input)
            # Apply mod after each multiplication to keep intermediate values manageable
            term = (self.coefficients[i + 1] * input_val) % self.large_prime
            result = (result + term) % self.large_prime

        # Final modulo with max_output (now int64) and return as int64
        # Don't cast back to int32 since max_output can be > int32 range
        return result % self.max_output
