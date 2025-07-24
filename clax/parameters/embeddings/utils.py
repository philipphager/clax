import jax
from flax import nnx

EIGHT_MERSENNE_PRIME = 2**31 - 1


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
        large_prime: int = EIGHT_MERSENNE_PRIME,
    ):
        super().__init__()
        self.large_prime = large_prime
        self.max_output = max_output
        self.num_args = num_args

        self.coefficients = jax.random.randint(
            rngs(),
            shape=(self.num_args + 1,),
            minval=1,
            maxval=self.large_prime,
        )
        self.coefficients = self.coefficients.at[0].set(
            jax.random.randint(
                rngs(),
                shape=(),
                minval=0,
                maxval=self.large_prime,
            )
        )

    def __call__(self, *hash_inputs):
        assert (
            len(hash_inputs) == self.num_args
        ), f"UniversalHash expects {self.num_args} arguments, but got {len(hash_inputs)}"

        # Start with constant term in range [0, P):
        result = self.coefficients[0]

        for i, hash_input in enumerate(hash_inputs):
            # Multiply a term in range [1, P) to each provided input:
            result += self.coefficients[i + 1] * hash_input

        return (result % self.large_prime) % self.max_output
